# CrisisQ device metrics

Local platform channel used by the CrisisQ study app. `CrisisqDeviceMetrics.snapshot()` returns the OS thermal state and current whole-app memory: Android Debug.getPss() in KiB, or iOS MACH_TASK_BASIC_INFO resident_size in bytes. Neither measurement is a peak or model-only allocation. No network access or user identifier is collected.
