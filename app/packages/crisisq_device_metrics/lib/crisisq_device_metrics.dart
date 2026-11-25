import 'package:flutter/services.dart';

class CrisisqDeviceMetrics {
  static const _channel = MethodChannel('crisisq/device_metrics');
  static Future<Map<String, dynamic>> snapshot() async {
    final data = await _channel.invokeMapMethod<String, dynamic>('snapshot');
    return Map<String, dynamic>.from(data ?? {});
  }
}
