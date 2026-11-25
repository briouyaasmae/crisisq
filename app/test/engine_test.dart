import 'dart:io';
import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:crisisq_device_study/models/study_packet.dart';
import 'package:crisisq_device_study/core/frozen_tokenizer.dart';
import 'package:crisisq_device_study/core/study_engine.dart';
import 'package:crisisq_device_study/services/native_model.dart';

class FakeModel implements LogitModel {
  final double probability;
  int calls = 0;
  FakeModel(this.probability);
  @override
  Future<List<double>> run(List<int> ids, List<int> mask) async {
    calls++;
    expect(ids.length, 128);
    expect(mask.length, 128);
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
  test('Hybrid exit skips both tokenizer and model', () async {
    final c = packet.cases.firstWhere((c) => (c['gate_v2'] as int) >= 0);
    final model = FakeModel(.5);
    final row = await engine.timedOne(
      input: c,
      variant: 'fp32',
      mode: 'hybrid',
      fullPipeline: true,
      model: model,
    );
    expect(model.calls, 0);
    expect(row['invoked'], false);
    expect(row['tokenization_ms'], 0);
    expect(row['onnx_ms'], null);
    expect(row['probability'], null);
    expect(row['decision'], c['gate_v2']);
  });
  test(
    'Neural execution preserves references with both tokenization scopes',
    () async {
      final c = packet.cases.first;
      final model = FakeModel(packet.probability(c, 'fp32'));
      for (final full in [true, false]) {
        final row = await engine.timedOne(
          input: c,
          variant: 'fp32',
          mode: 'neural',
          fullPipeline: full,
          model: model,
        );
        expect(row['parity_verified'], true);
        expect(row['invoked'], true);
        expect(row['tokenizer_mode'], full ? 'on_device' : 'pretokenized');
      }
      expect(model.calls, 2);
    },
  );
  test(
    'Changed native outputs fail closed instead of recording valid timings',
    () async {
      final c = packet.cases.first;
      final ref = packet.probability(c, 'fp32');
      final model = FakeModel(ref > 0.5 ? 0.1 : 0.9);
      await expectLater(
        engine.timedOne(
          input: c,
          variant: 'fp32',
          mode: 'neural',
          fullPipeline: true,
          model: model,
        ),
        throwsA(isA<ParityFailure>()),
      );
    },
  );
}
