package com.example.wizardondevice

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.abs

data class HandTrackingResult(
    val landmarks: FloatArray?,
    val rawResult: HandLandmarkerResult?,
    val trackerStatus: String,
    val inferenceTimeMs: Long,
    val detectedHandCount: Int,
    val selectedHandIndex: Int?,
)

class HandTracker(
    context: Context,
    assetName: String = HAND_LANDMARKER_ASSET,
) : AutoCloseable {
    private val handLandmarker: HandLandmarker
    private var lastSelectedLandmarks: List<NormalizedLandmark>? = null

    init {
        val baseOptions = BaseOptions.builder()
            .setDelegate(Delegate.CPU)
            .setModelAssetPath(assetName)
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(MAX_TRACKED_HANDS)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    fun detect(bitmap: Bitmap, timestampMs: Long): HandTrackingResult {
        val startedAt = SystemClock.uptimeMillis()
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = handLandmarker.detectForVideo(mpImage, timestampMs)
        val elapsed = SystemClock.uptimeMillis() - startedAt
        val candidateHands = result.landmarks()
        val selectedHandIndex = selectBestHandIndex(candidateHands)
        val handLandmarks = selectedHandIndex?.let(candidateHands::get)

        if (handLandmarks.isNullOrEmpty()) {
            return HandTrackingResult(
                landmarks = null,
                rawResult = result,
                trackerStatus = "No hand detected",
                inferenceTimeMs = elapsed,
                detectedHandCount = candidateHands.size,
                selectedHandIndex = null,
            )
        }

        lastSelectedLandmarks = handLandmarks

        val flattened = FloatArray(63)
        handLandmarks.forEachIndexed { index, landmark ->
            val offset = index * 3
            flattened[offset] = landmark.x()
            flattened[offset + 1] = landmark.y()
            flattened[offset + 2] = landmark.z()
        }

        return HandTrackingResult(
            landmarks = flattened,
            rawResult = result,
            trackerStatus = describeTrackingStatus(candidateHands.size, selectedHandIndex),
            inferenceTimeMs = elapsed,
            detectedHandCount = candidateHands.size,
            selectedHandIndex = selectedHandIndex,
        )
    }

    fun reset() {
        lastSelectedLandmarks = null
    }

    override fun close() {
        handLandmarker.close()
    }

    companion object {
        const val HAND_LANDMARKER_ASSET = "hand_landmarker.task"
        private const val MAX_TRACKED_HANDS = 3
        private const val TRACK_MATCH_THRESHOLD = 0.18f
        private const val WRIST_INDEX = 0
    }

    private fun selectBestHandIndex(candidateHands: List<List<NormalizedLandmark>>): Int? {
        if (candidateHands.isEmpty()) {
            lastSelectedLandmarks = null
            return null
        }
        if (candidateHands.size == 1) {
            return 0
        }

        val previous = lastSelectedLandmarks
        if (previous != null) {
            val closestIndex = candidateHands.indices.minByOrNull { index ->
                wristDistance(candidateHands[index], previous)
            }
            if (closestIndex != null) {
                val distance = wristDistance(candidateHands[closestIndex], previous)
                if (distance <= TRACK_MATCH_THRESHOLD) {
                    return closestIndex
                }
            }
        }

        return candidateHands.indices.maxByOrNull { index ->
            handProminenceScore(candidateHands[index])
        }
    }

    private fun wristDistance(
        currentHand: List<NormalizedLandmark>,
        previousHand: List<NormalizedLandmark>,
    ): Float {
        val currentWrist = currentHand[WRIST_INDEX]
        val previousWrist = previousHand[WRIST_INDEX]
        return abs(currentWrist.x() - previousWrist.x()) + abs(currentWrist.y() - previousWrist.y())
    }

    private fun handProminenceScore(hand: List<NormalizedLandmark>): Float {
        val minX = hand.minOf { it.x() }
        val maxX = hand.maxOf { it.x() }
        val minY = hand.minOf { it.y() }
        val maxY = hand.maxOf { it.y() }
        val area = (maxX - minX) * (maxY - minY)
        val centerX = (minX + maxX) * 0.5f
        val centerY = (minY + maxY) * 0.5f
        val centerPenalty = abs(centerX - 0.5f) + abs(centerY - 0.5f)
        return area - 0.1f * centerPenalty
    }

    private fun describeTrackingStatus(handCount: Int, selectedHandIndex: Int?): String {
        return if (handCount <= 1 || selectedHandIndex == null) {
            "Tracking hand"
        } else {
            "Tracking hand ${selectedHandIndex + 1}/$handCount"
        }
    }
}
