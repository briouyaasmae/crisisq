#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint crisisq_device_metrics.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'crisisq_device_metrics'
  s.version          = '1.0.0'
  s.summary          = 'Current process memory and thermal state for CrisisQ research.'
  s.description      = <<-DESC
Current process memory and thermal state for CrisisQ research.
                       DESC
  s.homepage         = 'https://flutter.dev'
  s.license          = { :file => '../LICENSE' }
  s.author           = 'CrisisQ project contributors'
  s.source           = { :path => '.' }
  s.source_files = 'crisisq_device_metrics/Sources/crisisq_device_metrics/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '16.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  # s.resource_bundles = {'crisisq_device_metrics_privacy' => ['crisisq_device_metrics/Sources/crisisq_device_metrics/PrivacyInfo.xcprivacy']}
end
