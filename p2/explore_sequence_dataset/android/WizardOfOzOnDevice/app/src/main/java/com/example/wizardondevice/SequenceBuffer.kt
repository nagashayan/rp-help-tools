package com.example.wizardondevice

class SequenceBuffer(
    private val windowSize: Int = 30,
    private val featureCount: Int = 63,
) {
    private val frames = ArrayDeque<FloatArray>(windowSize)

    fun append(frame: FloatArray) {
        require(frame.size == featureCount) {
            "Expected $featureCount features but found ${frame.size}"
        }
        if (frames.size == windowSize) {
            frames.removeFirst()
        }
        frames.addLast(frame.copyOf())
    }

    fun clear() {
        frames.clear()
    }

    fun size(): Int = frames.size

    fun isFull(): Boolean = frames.size == windowSize

    fun flattenForModel(): FloatArray {
        require(isFull()) { "Sequence buffer must be full before flattening." }
        val output = FloatArray(windowSize * featureCount)
        val snapshot = frames.toList()
        for (featureIndex in 0 until featureCount) {
            for (timeIndex in 0 until windowSize) {
                output[featureIndex * windowSize + timeIndex] = snapshot[timeIndex][featureIndex]
            }
        }
        return output
    }
}
