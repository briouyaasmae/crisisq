import Flutter
import UIKit
import Darwin

public class CrisisqDeviceMetricsPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "crisisq/device_metrics", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(CrisisqDeviceMetricsPlugin(), channel: channel)
  }
  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "snapshot" else { result(FlutterMethodNotImplemented); return }
    var info = mach_task_basic_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
    let capacity = Int(count)
    let status = withUnsafeMutablePointer(to: &info) { ptr in
      ptr.withMemoryRebound(to: integer_t.self, capacity: capacity) { rebound in
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
      }
    }
    let thermal: String
    switch ProcessInfo.processInfo.thermalState {
      case .nominal: thermal = "nominal"
      case .fair: thermal = "fair"
      case .serious: thermal = "serious"
      case .critical: thermal = "critical"
      @unknown default: thermal = "unknown"
    }
    let resident: Any = status == KERN_SUCCESS ? NSNumber(value: UInt64(info.resident_size)) : NSNull()
    result([
      "thermal_state": thermal,
      "memory_metric": "ios_process_resident_bytes",
      "memory_value": resident,
      "memory_scope": "whole_app_current_snapshot_not_peak",
      "platform": "ios"
    ])
  }
}
