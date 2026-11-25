import 'dart:io';
import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:crisisq_device_study/core/frozen_tokenizer.dart';
import 'package:crisisq_device_study/core/study_engine.dart';
import 'package:crisisq_device_study/models/study_packet.dart';
import 'package:crisisq_device_study/services/native_model.dart';
import 'package:crisisq_device_study/services/study_controller.dart';

class ProbabilityModel implements LogitModel {
  final double probability;
  int calls = 0;
  ProbabilityModel(this.probability);
  @override
  Future<List<double>> run(List<int> ids, List<int> mask) async {
    calls++;
    return [log(1 - probability), log(probability)];
  }

  @override
  Future<void> close() async {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late StudyPacket packet;
  late StudyEngine engine;
  setUpAll(() async {
    packet = await StudyPacket.load();
    engine = StudyEngine(
      packet,
      FrozenBertTokenizer.fromJson(
        File('test/fixtures/base_tokenizer_reference.json').readAsStringSync(),
      ),
    );
  });
  test(
    'Portability explicitly retains Kaggle failures while checking device output',
    () async {
      final input = packet.cases.first;
      final ref = packet.probability(input, 'q_quint8');
      final p = ref > 0.5 ? 0.1 : 0.9;
      final row = await engine.timedOne(
        input: input,
        variant: 'q_quint8',
        mode: 'neural',
        fullPipeline: true,
        model: ProbabilityModel(p),
        requireKaggleParity: false,
        deviceReferenceProbability: p,
      );
      expect(row['parity_verified'], false);
      expect(row['kaggle_probability_match'], false);
      expect(row['device_repeatability_verified'], true);
      expect(row['neural_reference_probability'], ref);
      expect(
        row['validation_scope'],
        'device_repeatability_with_recorded_kaggle_drift',
      );
      expect(row['decision'], p >= packet.tau ? 1 : 0);
    },
  );
  test('Portability rejects within-device changes', () async {
    await expectLater(
      engine.timedOne(
        input: packet.cases.first,
        variant: 'q_quint8',
        mode: 'neural',
        fullPipeline: false,
        model: ProbabilityModel(0.9),
        requireKaggleParity: false,
        deviceReferenceProbability: 0.1,
      ),
      throwsA(isA<ParityFailure>()),
    );
  });
  test('Portability rejects missing or nonfinite device reference', () async {
    for (final value in [null, double.nan]) {
      await expectLater(
        engine.timedOne(
          input: packet.cases.first,
          variant: 'q_quint8',
          mode: 'neural',
          fullPipeline: false,
          model: ProbabilityModel(0.9),
          requireKaggleParity: false,
          deviceReferenceProbability: value,
        ),
        throwsA(isA<ParityFailure>()),
      );
    }
  });
  test('FP32 cannot use the exploratory reference policy', () async {
    await expectLater(
      engine.timedOne(
        input: packet.cases.first,
        variant: 'fp32',
        mode: 'neural',
        fullPipeline: false,
        model: ProbabilityModel(0.9),
        requireKaggleParity: false,
        deviceReferenceProbability: 0.9,
      ),
      throwsA(isA<ArgumentError>()),
    );
  });
  test('Portability still skips native execution for a gate exit', () async {
    final input = packet.cases.firstWhere((c) => c['gate_v2'] >= 0);
    final model = ProbabilityModel(0.9);
    final row = await engine.timedOne(
      input: input,
      variant: 'q_quint8',
      mode: 'hybrid',
      fullPipeline: true,
      model: model,
      requireKaggleParity: false,
      deviceReferenceProbability: 0.9,
    );
    expect(model.calls, 0);
    expect(row['device_repeatability_verified'], null);
    expect(row['probability'], null);
    expect(row['decision'], input['gate_v2']);
  });
  test(
    'Complete QUInt8 outputs plus explicit mode permit measurement, not strict parity',
    () {
      final c = StudyController()..packet = packet;
      c.configure(onlyFp32: false);
      c.variantPassed.addAll({'fp32': true, 'q_quint8': false});
      c.parityReport = {
        'models': [
          for (final input in packet.cases)
            {
              'variant': 'q_quint8',
              'source_id': input['source_id'],
              'actual_probability': 0.5,
            },
        ],
      };
      expect(c.selectedModelsReady, false);
      c.configure(permitQuint8Drift: true);
      expect(c.selectedModelsReady, true);
      expect(c.selectedReferencesPassed, false);
      expect(c.canBenchmark, false, reason: 'Debug mode must remain blocked');
      c.configure(threadCount: 2);
      expect(c.quint8DeviceReferencesReady, false);
      expect(c.selectedModelsReady, false);
    },
  );
  test('Incomplete QUInt8 reference collection cannot enable portability', () {
    final c = StudyController()..packet = packet;
    c.configure(onlyFp32: false, permitQuint8Drift: true);
    c.variantPassed['fp32'] = true;
    c.parityReport = {
      'models': [
        for (final input in packet.cases.skip(1))
          {
            'variant': 'q_quint8',
            'source_id': input['source_id'],
            'actual_probability': 0.5,
          },
      ],
    };
    expect(c.selectedModelsReady, false);
  });
}
