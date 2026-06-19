package com.example.wizardondevice

import org.junit.Assert.assertEquals
import org.junit.Test

class SequenceBufferTest {
    @Test
    fun flattenForModel_usesFeatureMajorOrdering() {
        val buffer = SequenceBuffer(windowSize = 2, featureCount = 3)
        buffer.append(floatArrayOf(1f, 2f, 3f))
        buffer.append(floatArrayOf(4f, 5f, 6f))

        val flattened = buffer.flattenForModel()

        assertEquals(listOf(1f, 4f, 2f, 5f, 3f, 6f), flattened.toList())
    }
}
