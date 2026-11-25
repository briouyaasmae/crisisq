import 'package:flutter_test/flutter_test.dart';
import 'package:crisisq_device_study/services/study_controller.dart';

void main() {
  test('FP32-only mode accepts passed FP32 without accepting failed INT8', () {
    final c = StudyController();
    c.variantPassed.addAll({'fp32': true, 'q_quint8': false});
    expect(c.selectedReferencesPassed, true);
    expect(c.selectedVariants, ['fp32']);
    expect(c.studyTaskCount, 8);
    expect(c.studyMeasurements, 2560);
    c.configure(onlyFp32: false);
    expect(c.selectedReferencesPassed, false);
    expect(c.studyTaskCount, 16);
    expect(c.studyMeasurements, 5120);
    c.variantPassed['q_quint8'] = true;
    expect(c.selectedReferencesPassed, true);
  });
  test('Missing or failed FP32 still blocks selected-model validation', () {
    final c = StudyController();
    expect(c.selectedReferencesPassed, false);
    c.variantPassed.addAll({'fp32': false, 'q_quint8': true});
    expect(c.selectedReferencesPassed, false);
  });
  test('Thread change invalidates every selected model check', () {
    final c = StudyController();
    c.variantPassed['fp32'] = true;
    expect(c.selectedReferencesPassed, true);
    c.configure(threadCount: 2);
    expect(c.variantPassed, isEmpty);
    expect(c.selectedReferencesPassed, false);
  });
  test('Passing references never bypasses debug build restriction', () {
    final c = StudyController();
    c.modelsReady = true;
    c.tokenizerPassed = true;
    c.physicalDevice = true;
    c.variantPassed['fp32'] = true;
    expect(c.canBenchmark, false);
  });
}
