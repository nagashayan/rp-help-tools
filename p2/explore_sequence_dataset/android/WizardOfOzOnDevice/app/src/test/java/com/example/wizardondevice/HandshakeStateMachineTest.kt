package com.example.wizardondevice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HandshakeStateMachineTest {
    @Test
    fun update_holdsStateAndCooldownsVibration() {
        val machine = HandshakeStateMachine(threshold = 0.8f, holdMillis = 500L, vibrationCooldownMillis = 1000L)

        val first = machine.update(probabilityClass1 = 0.9f, nowMs = 100L)
        assertEquals(HandshakeState.HANDSHAKE, first.state)
        assertTrue(first.shouldVibrate)

        val second = machine.update(probabilityClass1 = 0.95f, nowMs = 200L)
        assertEquals(HandshakeState.HANDSHAKE, second.state)
        assertFalse(second.shouldVibrate)

        val held = machine.update(probabilityClass1 = 0.2f, nowMs = 400L)
        assertEquals(HandshakeState.HANDSHAKE, held.state)

        val ended = machine.update(probabilityClass1 = 0.2f, nowMs = 700L)
        assertEquals(HandshakeState.IDLE, ended.state)
    }
}
