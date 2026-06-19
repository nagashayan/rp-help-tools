package com.example.wizardondevice

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private val uiScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val frameSource = MacFrameSource()
    private val sequenceBuffer = SequenceBuffer()
    private val stateMachine = HandshakeStateMachine()

    private lateinit var ipInput: EditText
    private lateinit var connectButton: Button
    private lateinit var pocketModeButton: Button
    private lateinit var previewImage: ImageView
    private lateinit var overlayView: OverlayView
    private lateinit var stateText: TextView
    private lateinit var connectionText: TextView
    private lateinit var trackerText: TextView
    private lateinit var bufferText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var pocketModeOverlay: FrameLayout
    private lateinit var pocketModeStatus: TextView

    private var pipelineJob: Job? = null
    private var classifier: HandshakeClassifier? = null
    private var handTracker: HandTracker? = null
    private var lastValidLandmarks: FloatArray? = null
    private var trackingGapFrames = 0
    private var isVibrating = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()
        bindActions()
        renderDisconnectedState()
    }

    private fun bindViews() {
        ipInput = findViewById(R.id.ipInput)
        connectButton = findViewById(R.id.connectButton)
        pocketModeButton = findViewById(R.id.pocketModeButton)
        previewImage = findViewById(R.id.previewImage)
        overlayView = findViewById(R.id.overlayView)
        stateText = findViewById(R.id.stateText)
        connectionText = findViewById(R.id.connectionText)
        trackerText = findViewById(R.id.trackerText)
        bufferText = findViewById(R.id.bufferText)
        confidenceText = findViewById(R.id.confidenceText)
        inferenceText = findViewById(R.id.inferenceText)
        pocketModeOverlay = findViewById(R.id.pocketModeOverlay)
        pocketModeStatus = findViewById(R.id.pocketModeStatus)
    }

    private fun bindActions() {
        connectButton.setOnClickListener {
            if (pipelineJob?.isActive == true) {
                stopPipeline()
            } else {
                val host = ipInput.text.toString().trim()
                if (host.isNotEmpty()) {
                    startPipeline(host)
                } else {
                    connectionText.text = "Enter the Mac IP address."
                    connectionText.setTextColor(getColor(R.color.warning))
                }
            }
        }

        pocketModeButton.setOnClickListener {
            pocketModeOverlay.visibility = View.VISIBLE
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            val params = window.attributes
            params.screenBrightness = 0.0f
            window.attributes = params
        }

        pocketModeOverlay.setOnLongClickListener {
            pocketModeOverlay.visibility = View.GONE
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            val params = window.attributes
            params.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
            window.attributes = params
            true
        }
    }

    private fun startPipeline(host: String) {
        renderConnectingState()
        pipelineJob?.cancel()
        pipelineJob = uiScope.launch(Dispatchers.IO) {
            try {
                ensurePipelineComponents()
                val healthy = frameSource.checkHealth(host)
                if (!healthy) {
                    withContext(Dispatchers.Main) {
                        renderConnectionFailure("Mac camera server is not ready.")
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    connectButton.text = "Disconnect"
                    connectButton.isEnabled = true
                    pocketModeButton.visibility = View.VISIBLE
                    pocketModeButton.isEnabled = true
                    connectionText.text = "Connected to $host"
                    connectionText.setTextColor(getColor(R.color.accent))
                }

                while (isActive) {
                    val startedAt = SystemClock.uptimeMillis()
                    val bitmap = frameSource.fetchSnapshot(host)
                    if (bitmap == null) {
                        withContext(Dispatchers.Main) {
                            connectionText.text = "Snapshot fetch failed."
                            connectionText.setTextColor(getColor(R.color.error))
                        }
                        delay(250)
                        continue
                    }

                    val trackingResult = handTracker!!.detect(bitmap, startedAt)
                    val reusedLandmarks = handleTrackingResult(trackingResult)
                    val classifierStartedAt = SystemClock.uptimeMillis()
                    val classifierResult = if (sequenceBuffer.isFull()) {
                        classifier!!.classify(sequenceBuffer.flattenForModel())
                    } else {
                        ClassificationResult(logits = floatArrayOf(0f, 0f), probabilityClass1 = 0f)
                    }
                    val classifierInferenceMs = if (sequenceBuffer.isFull()) {
                        SystemClock.uptimeMillis() - classifierStartedAt
                    } else {
                        0L
                    }

                    val stateUpdate = stateMachine.update(
                        probabilityClass1 = classifierResult.probabilityClass1,
                        nowMs = SystemClock.uptimeMillis(),
                    )

                    withContext(Dispatchers.Main) {
                        renderFrame(
                            bitmap = bitmap,
                            trackingResult = trackingResult,
                            probability = classifierResult.probabilityClass1,
                            state = stateUpdate.state,
                            classifierInferenceMs = classifierInferenceMs,
                            totalInferenceMs = trackingResult.inferenceTimeMs + (SystemClock.uptimeMillis() - startedAt),
                            reusedLandmarks = reusedLandmarks,
                        )
                        if (stateUpdate.shouldVibrate) {
                            triggerHaptics()
                        }
                    }

                    val elapsed = SystemClock.uptimeMillis() - startedAt
                    val remaining = FRAME_INTERVAL_MS - elapsed
                    if (remaining > 0) {
                        delay(remaining)
                    }
                }
            } catch (error: Exception) {
                withContext(Dispatchers.Main) {
                    renderConnectionFailure(error.message ?: "Pipeline stopped unexpectedly.")
                }
            }
        }
    }

    private fun stopPipeline() {
        pipelineJob?.cancel()
        pipelineJob = null
        sequenceBuffer.clear()
        stateMachine.reset()
        lastValidLandmarks = null
        trackingGapFrames = 0
        handTracker?.reset()
        overlayView.clear()
        renderDisconnectedState()
    }

    private suspend fun ensurePipelineComponents() {
        if (classifier == null) {
            classifier = HandshakeClassifier(applicationContext)
        }
        if (handTracker == null) {
            handTracker = HandTracker(applicationContext)
        }
    }

    private fun handleTrackingResult(trackingResult: HandTrackingResult): Boolean {
        trackingResult.landmarks?.let { landmarks ->
            trackingGapFrames = 0
            lastValidLandmarks = landmarks
            sequenceBuffer.append(landmarks)
            return false
        }

        if (lastValidLandmarks != null && trackingGapFrames < MAX_TRACKING_GAP_FRAMES) {
            trackingGapFrames += 1
            sequenceBuffer.append(lastValidLandmarks!!)
            return true
        }

        trackingGapFrames = MAX_TRACKING_GAP_FRAMES
        lastValidLandmarks = null
        sequenceBuffer.clear()
        return false
    }

    private fun renderFrame(
        bitmap: Bitmap,
        trackingResult: HandTrackingResult,
        probability: Float,
        state: HandshakeState,
        classifierInferenceMs: Long,
        totalInferenceMs: Long,
        reusedLandmarks: Boolean,
    ) {
        previewImage.setImageBitmap(bitmap)
        val rawResult = trackingResult.rawResult
        if (rawResult != null && rawResult.landmarks().isNotEmpty()) {
            overlayView.setResults(rawResult, bitmap.height, bitmap.width)
        } else {
            overlayView.clear()
        }

        val stateColor = when (state) {
            HandshakeState.HANDSHAKE -> getColor(R.color.success)
            HandshakeState.IDLE -> getColor(R.color.text_primary)
        }

        stateText.text = state.name
        stateText.setTextColor(stateColor)
        pocketModeStatus.text = state.name
        pocketModeStatus.setTextColor(stateColor)

        trackerText.text = if (reusedLandmarks) {
            "Tracker: gap fill ($trackingGapFrames/$MAX_TRACKING_GAP_FRAMES)"
        } else {
            "Tracker: ${trackingResult.trackerStatus} | hands: ${trackingResult.detectedHandCount}"
        }
        bufferText.text = "Buffer: ${sequenceBuffer.size()}/30"
        confidenceText.text = "Confidence: ${"%.1f".format(probability * 100f)}%"
        inferenceText.text =
            "Latency: tracker ${trackingResult.inferenceTimeMs} ms | classifier ${classifierInferenceMs} ms | total ${totalInferenceMs} ms"
        connectionText.text = "Polling snapshots locally"
        connectionText.setTextColor(getColor(R.color.text_secondary))
    }

    private fun renderConnectingState() {
        connectButton.text = "Connecting..."
        connectButton.isEnabled = false
        pocketModeButton.visibility = View.VISIBLE
        pocketModeButton.isEnabled = false
        connectionText.text = "Checking camera server..."
        connectionText.setTextColor(getColor(R.color.warning))
        trackerText.text = "Tracker: waiting"
        bufferText.text = "Buffer: 0/30"
        confidenceText.text = "Confidence: --"
        inferenceText.text = "Frame latency: --"
        stateText.text = HandshakeState.IDLE.name
        stateText.setTextColor(getColor(R.color.text_primary))
        pocketModeStatus.text = HandshakeState.IDLE.name
    }

    private fun renderDisconnectedState() {
        connectButton.text = "Connect"
        connectButton.isEnabled = true
        pocketModeButton.visibility = View.GONE
        connectionText.text = "Waiting for Mac server"
        connectionText.setTextColor(getColor(R.color.text_secondary))
        trackerText.text = "Tracker: idle"
        bufferText.text = "Buffer: 0/30"
        confidenceText.text = "Confidence: --"
        inferenceText.text = "Frame latency: --"
        stateText.text = HandshakeState.IDLE.name
        stateText.setTextColor(getColor(R.color.text_primary))
        pocketModeStatus.text = HandshakeState.IDLE.name
        pocketModeStatus.setTextColor(getColor(R.color.text_primary))
    }

    private fun renderConnectionFailure(message: String) {
        connectButton.text = "Connect"
        connectButton.isEnabled = true
        pocketModeButton.isEnabled = false
        connectionText.text = message
        connectionText.setTextColor(getColor(R.color.error))
        trackerText.text = "Tracker: unavailable"
        stateText.text = HandshakeState.IDLE.name
        stateText.setTextColor(getColor(R.color.text_primary))
        pocketModeStatus.text = HandshakeState.IDLE.name
    }

    private fun triggerHaptics() {
        if (isVibrating) return
        isVibrating = true
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val manager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            manager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        if (vibrator.hasVibrator()) {
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 180, 100, 180), -1))
        }

        uiScope.launch {
            delay(600)
            isVibrating = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipelineJob?.cancel()
        classifier?.close()
        handTracker?.close()
        uiScope.cancel()
    }

    companion object {
        private const val FRAME_INTERVAL_MS = 67L
        private const val MAX_TRACKING_GAP_FRAMES = 3
    }
}
