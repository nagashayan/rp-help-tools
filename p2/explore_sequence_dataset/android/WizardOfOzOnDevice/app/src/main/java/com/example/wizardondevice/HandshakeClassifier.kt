package com.example.wizardondevice

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

data class ClassificationResult(
    val logits: FloatArray,
    val probabilityClass1: Float,
)

class HandshakeClassifier(
    context: Context,
    modelAssetName: String = MODEL_ASSET_NAME,
) : AutoCloseable {
    private val interpreter = Interpreter(
        loadModelFile(context, modelAssetName),
        Interpreter.Options().apply {
            setNumThreads(2)
            setUseXNNPACK(true)
        }
    )

    fun classify(input: FloatArray): ClassificationResult {
        require(input.size == INPUT_ELEMENT_COUNT) {
            "Expected $INPUT_ELEMENT_COUNT floats but found ${input.size}"
        }

        val inputBuffer = ByteBuffer.allocateDirect(INPUT_ELEMENT_COUNT * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        input.forEach(inputBuffer::putFloat)
        inputBuffer.rewind()

        val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_ELEMENT_COUNT * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val logits = FloatArray(OUTPUT_ELEMENT_COUNT)
        outputBuffer.asFloatBuffer().get(logits)

        return ClassificationResult(
            logits = logits,
            probabilityClass1 = softmaxClass1(logits),
        )
    }

    override fun close() {
        interpreter.close()
    }

    private fun softmaxClass1(logits: FloatArray): Float {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exp0 = exp((logits[0] - maxLogit).toDouble())
        val exp1 = exp((logits[1] - maxLogit).toDouble())
        return (exp1 / (exp0 + exp1)).toFloat()
    }

    companion object {
        const val MODEL_ASSET_NAME = "temporal_cnn_raw.tflite"
        private const val INPUT_ELEMENT_COUNT = 63 * 30
        private const val OUTPUT_ELEMENT_COUNT = 2

        private fun loadModelFile(context: Context, modelAssetName: String): MappedByteBuffer {
            val fileDescriptor = context.assets.openFd(modelAssetName)
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                return inputStream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength,
                )
            }
        }
    }
}
