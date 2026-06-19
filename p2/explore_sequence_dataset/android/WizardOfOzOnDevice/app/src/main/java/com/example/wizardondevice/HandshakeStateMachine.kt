package com.example.wizardondevice

data class StateUpdate(
    val state: HandshakeState,
    val shouldVibrate: Boolean,
)

class HandshakeStateMachine(
    private val threshold: Float = 0.80f,
    private val holdMillis: Long = 500L,
    private val vibrationCooldownMillis: Long = 1_500L,
) {
    private var holdUntilMs: Long = 0L
    private var nextAllowedVibrationMs: Long = 0L

    fun update(probabilityClass1: Float, nowMs: Long): StateUpdate {
        var shouldVibrate = false
        if (probabilityClass1 > threshold) {
            holdUntilMs = maxOf(holdUntilMs, nowMs + holdMillis)
            if (nowMs >= nextAllowedVibrationMs) {
                shouldVibrate = true
                nextAllowedVibrationMs = nowMs + vibrationCooldownMillis
            }
        }
        val state = if (nowMs < holdUntilMs) HandshakeState.HANDSHAKE else HandshakeState.IDLE
        return StateUpdate(state = state, shouldVibrate = shouldVibrate)
    }

    fun reset() {
        holdUntilMs = 0L
        nextAllowedVibrationMs = 0L
    }
}
