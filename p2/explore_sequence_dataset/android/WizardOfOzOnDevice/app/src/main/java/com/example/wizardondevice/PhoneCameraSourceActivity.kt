package com.example.wizardondevice

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.graphics.YuvImage
import android.hardware.Camera
import android.os.Bundle
import android.os.SystemClock
import android.view.Surface
import android.view.TextureView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.util.Collections
import java.util.concurrent.atomic.AtomicBoolean

@Suppress("DEPRECATION")
class PhoneCameraSourceActivity : AppCompatActivity(), TextureView.SurfaceTextureListener {
    private val uiScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private lateinit var previewView: TextureView
    private lateinit var statusText: TextView
    private lateinit var endpointText: TextView
    private lateinit var frameText: TextView

    private var camera: Camera? = null
    private var previewSize: Camera.Size? = null
    private var previewRotationDegrees: Int = 0
    private var serverJob: Job? = null
    private var serverSocket: ServerSocket? = null
    private val isEncodingFrame = AtomicBoolean(false)

    @Volatile
    private var latestJpeg: ByteArray? = null

    @Volatile
    private var latestFrameAgeMs: Long = 0L

    private var lastEncodedAtMs: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_phone_camera_source)

        previewView = findViewById(R.id.sourcePreview)
        statusText = findViewById(R.id.sourceStatusText)
        endpointText = findViewById(R.id.sourceEndpointText)
        frameText = findViewById(R.id.sourceFrameText)

        previewView.surfaceTextureListener = this
        runCatching { refreshEndpointText() }
            .onFailure { endpointText.text = "Unable to determine local IP yet." }
    }

    override fun onResume() {
        super.onResume()
        startServer()
        if (hasCameraPermission() && previewView.isAvailable) {
            startCamera(previewView.surfaceTexture!!)
        } else if (!hasCameraPermission()) {
            requestCameraPermission()
        }
    }

    override fun onPause() {
        super.onPause()
        stopCamera()
        stopServer()
    }

    override fun onDestroy() {
        super.onDestroy()
        uiScope.cancel()
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        if (hasCameraPermission()) {
            startCamera(surface)
        } else {
            requestCameraPermission()
        }
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        stopCamera()
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            if (previewView.isAvailable) {
                startCamera(previewView.surfaceTexture!!)
            }
        } else {
            statusText.text = "Camera permission is required for source mode."
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST)
    }

    private fun startCamera(surfaceTexture: SurfaceTexture) {
        if (camera != null) {
            return
        }

        try {
            val cameraInstance = Camera.open(findBackCameraId())
            val parameters = cameraInstance.parameters
            val selectedSize = choosePreviewSize(parameters.supportedPreviewSizes)
            parameters.setPreviewSize(selectedSize.width, selectedSize.height)
            parameters.previewFormat = ImageFormat.NV21
            if (parameters.supportedFocusModes?.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO) == true) {
                parameters.focusMode = Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO
            }
            cameraInstance.parameters = parameters

            previewSize = selectedSize
            previewRotationDegrees = computePreviewRotation()
            cameraInstance.setDisplayOrientation(previewRotationDegrees)
            cameraInstance.setPreviewTexture(surfaceTexture)

            val bufferSize = selectedSize.width * selectedSize.height * ImageFormat.getBitsPerPixel(ImageFormat.NV21) / 8
            cameraInstance.addCallbackBuffer(ByteArray(bufferSize))
            cameraInstance.setPreviewCallbackWithBuffer { data, activeCamera ->
                handlePreviewFrame(data, activeCamera)
                activeCamera.addCallbackBuffer(data)
            }

            cameraInstance.startPreview()
            camera = cameraInstance
            statusText.text = "Camera source is live."
            frameText.text = "Waiting for first frame..."
        } catch (error: Exception) {
            statusText.text = "Camera failed: ${error.message ?: "unknown error"}"
        }
    }

    private fun stopCamera() {
        camera?.apply {
            setPreviewCallbackWithBuffer(null)
            stopPreview()
            release()
        }
        camera = null
        previewSize = null
        latestJpeg = null
        latestFrameAgeMs = 0L
    }

    private fun handlePreviewFrame(data: ByteArray, activeCamera: Camera) {
        val size = previewSize ?: return
        val now = SystemClock.uptimeMillis()
        if (now - lastEncodedAtMs < MIN_FRAME_ENCODE_INTERVAL_MS) {
            return
        }
        if (!isEncodingFrame.compareAndSet(false, true)) {
            return
        }
        lastEncodedAtMs = now

        val frameCopy = data.copyOf()
        uiScope.launch(Dispatchers.Default) {
            try {
                val jpegBytes = encodePreviewFrame(frameCopy, size.width, size.height, previewRotationDegrees)
                latestJpeg = jpegBytes
                latestFrameAgeMs = SystemClock.uptimeMillis() - now
                withContext(Dispatchers.Main) {
                    frameText.text = "Serving ${size.width}x${size.height} JPEGs on port $SERVER_PORT | encode ${latestFrameAgeMs} ms"
                }
            } finally {
                isEncodingFrame.set(false)
            }
        }
    }

    private fun encodePreviewFrame(
        nv21: ByteArray,
        width: Int,
        height: Int,
        rotationDegrees: Int,
    ): ByteArray {
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val jpegStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), JPEG_QUALITY, jpegStream)
        val jpegBytes = jpegStream.toByteArray()

        if (rotationDegrees == 0) {
            return jpegBytes
        }

        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        val rotatedStream = ByteArrayOutputStream()
        rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, rotatedStream)
        if (rotatedBitmap != bitmap) {
            bitmap.recycle()
        }
        rotatedBitmap.recycle()
        return rotatedStream.toByteArray()
    }

    private fun startServer() {
        if (serverJob?.isActive == true) {
            return
        }

        serverJob = uiScope.launch(Dispatchers.IO) {
            try {
                val socket = ServerSocket(SERVER_PORT)
                serverSocket = socket
                withContext(Dispatchers.Main) {
                    refreshEndpointText()
                }

                while (isActive) {
                    val client = socket.accept()
                    launch {
                        handleClient(client)
                    }
                }
            } catch (error: Exception) {
                withContext(Dispatchers.Main) {
                    statusText.text = "Server failed: ${error.message ?: "unknown error"}"
                }
            }
        }
    }

    private fun stopServer() {
        serverSocket?.close()
        serverSocket = null
        serverJob?.cancel()
        serverJob = null
    }

    private fun handleClient(client: Socket) {
        client.use { socket ->
            socket.soTimeout = 1_000
            val input = socket.getInputStream().bufferedReader()
            val requestLine = input.readLine() ?: return
            while (true) {
                val header = input.readLine() ?: break
                if (header.isBlank()) break
            }

            val path = requestLine.split(" ").getOrNull(1) ?: "/"
            val output = BufferedOutputStream(socket.getOutputStream())
            when (path.substringBefore("?")) {
                "/health" -> writeHttpResponse(output, "text/plain", "ok".toByteArray())
                "/snapshot" -> {
                    val frame = latestJpeg
                    if (frame == null) {
                        writeHttpResponse(output, "text/plain", "warming up".toByteArray(), statusCode = 503)
                    } else {
                        writeHttpResponse(output, "image/jpeg", frame)
                    }
                }
                else -> writeHttpResponse(output, "text/plain", "not found".toByteArray(), statusCode = 404)
            }
        }
    }

    private fun writeHttpResponse(
        output: BufferedOutputStream,
        contentType: String,
        body: ByteArray,
        statusCode: Int = 200,
    ) {
        val reason = when (statusCode) {
            200 -> "OK"
            404 -> "Not Found"
            503 -> "Service Unavailable"
            else -> "OK"
        }
        output.write("HTTP/1.1 $statusCode $reason\r\n".toByteArray())
        output.write("Connection: close\r\n".toByteArray())
        output.write("Content-Type: $contentType\r\n".toByteArray())
        output.write("Content-Length: ${body.size}\r\n".toByteArray())
        output.write("Cache-Control: no-store\r\n".toByteArray())
        output.write("\r\n".toByteArray())
        output.write(body)
        output.flush()
    }

    private fun refreshEndpointText() {
        val ipAddress = runCatching { findSiteLocalIpAddress() }.getOrNull()
        endpointText.text = if (ipAddress == null) {
            "Connect this phone to Wi-Fi to expose http://<ip>:$SERVER_PORT/"
        } else {
            "Share this source endpoint with the inference phone:\nhttp://$ipAddress:$SERVER_PORT"
        }
    }

    private fun findSiteLocalIpAddress(): String? {
        val interfaces = NetworkInterface.getNetworkInterfaces() ?: return null
        return Collections.list(interfaces)
            .flatMap { Collections.list(it.inetAddresses) }
            .firstOrNull { address ->
                !address.isLoopbackAddress && address is Inet4Address && address.isSiteLocalAddress
            }
            ?.hostAddress
    }

    private fun choosePreviewSize(sizes: List<Camera.Size>): Camera.Size {
        return sizes
            .sortedBy { kotlin.math.abs(it.width - TARGET_PREVIEW_WIDTH) + kotlin.math.abs(it.height - TARGET_PREVIEW_HEIGHT) }
            .first()
    }

    private fun findBackCameraId(): Int {
        val cameraInfo = Camera.CameraInfo()
        for (index in 0 until Camera.getNumberOfCameras()) {
            Camera.getCameraInfo(index, cameraInfo)
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                return index
            }
        }
        return 0
    }

    private fun computePreviewRotation(): Int {
        val cameraInfo = Camera.CameraInfo()
        Camera.getCameraInfo(findBackCameraId(), cameraInfo)
        val rotation = display?.rotation ?: Surface.ROTATION_0
        val displayRotationDegrees = when (rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (cameraInfo.orientation - displayRotationDegrees + 360) % 360
    }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 1001
        private const val SERVER_PORT = 5000
        private const val TARGET_PREVIEW_WIDTH = 640
        private const val TARGET_PREVIEW_HEIGHT = 480
        private const val JPEG_QUALITY = 80
        private const val MIN_FRAME_ENCODE_INTERVAL_MS = 100L
    }
}
