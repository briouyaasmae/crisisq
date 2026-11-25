package org.crisisq.crisisq_device_metrics

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

class CrisisqDeviceMetricsPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        context = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, "crisisq/device_metrics")
        channel.setMethodCallHandler(this)
    }
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "snapshot") { result.notImplemented(); return }
        val manager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) manager.currentThermalStatus else -1
        val names = mapOf(0 to "none", 1 to "light", 2 to "moderate", 3 to "severe", 4 to "critical", 5 to "emergency", 6 to "shutdown")
        result.success(mapOf(
            "thermal_state" to (names[thermal] ?: "unavailable"),
            "thermal_code" to thermal,
            "memory_metric" to "android_process_pss_kib",
            "memory_value" to Debug.getPss(),
            "memory_scope" to "whole_app_current_snapshot_not_peak",
            "platform" to "android"
        ))
    }
    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) { channel.setMethodCallHandler(null) }
}
