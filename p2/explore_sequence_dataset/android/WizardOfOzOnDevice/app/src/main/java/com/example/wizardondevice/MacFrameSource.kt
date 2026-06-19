package com.example.wizardondevice

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.BufferedInputStream
import java.net.HttpURLConnection
import java.net.URL

class MacFrameSource(
    private val connectTimeoutMs: Int = 1_000,
    private val readTimeoutMs: Int = 1_000,
) {
    fun checkHealth(host: String): Boolean {
        val connection = openConnection(host, "health")
        return try {
            connection.responseCode == HttpURLConnection.HTTP_OK
        } finally {
            connection.disconnect()
        }
    }

    fun fetchSnapshot(host: String): Bitmap? {
        val connection = openConnection(host, "snapshot")
        return try {
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                null
            } else {
                BufferedInputStream(connection.inputStream).use { stream ->
                    BitmapFactory.decodeStream(stream)?.let { bitmap ->
                        if (bitmap.config == Bitmap.Config.ARGB_8888) {
                            bitmap
                        } else {
                            bitmap.copy(Bitmap.Config.ARGB_8888, false)
                        }
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun openConnection(host: String, path: String): HttpURLConnection {
        val url = URL("http://$host:5000/$path")
        return (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = connectTimeoutMs
            readTimeout = readTimeoutMs
            doInput = true
        }
    }
}
