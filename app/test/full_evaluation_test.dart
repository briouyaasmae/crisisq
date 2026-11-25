import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:crisisq_device_study/models/study_packet.dart';
import 'package:crisisq_device_study/models/full_evaluation_packet.dart';
import 'package:crisisq_device_study/core/frozen_tokenizer.dart';
import 'package:crisisq_device_study/core/study_engine.dart';
import 'package:crisisq_device_study/services/study_controller.dart';
import 'package:crisisq_device_study/services/full_evaluation_controller.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late StudyPacket study;
  late FullEvaluationPacket full;
  setUpAll(() async {
    study = await StudyPacket.load();
    full = await FullEvaluationPacket.load(study);
  });
  test('Exact 1700 labeled membership and class counts are retained', () {
    expect(full.cases.length, 1700);
    expect(full.cases.where((r) => r['label'] == 1).length, 800);
    expect(full.cases.where((r) => r['label'] == 0).length, 900);
    expect(full.cases.map((r) => r['source_id']).toSet().length, 1700);
  });
  test(
    'On-device tokenizer matches independent Rust token IDs on all 1700 inputs',
    () {
      final t = FrozenBertTokenizer.fromJson(
        File('test/fixtures/base_tokenizer_reference.json').readAsStringSync(),
      );
      final mismatches = <String>[];
      for (final row in full.cases) {
        final e = t.encode(row['text'] as String, 128);
        if (!sameInts(e.inputIds, (row['input_ids'] as List).cast<int>()) ||
            !sameInts(
              e.attentionMask,
              (row['attention_mask'] as List).cast<int>(),
            )) {
          mismatches.add(row['source_id'] as String);
        }
      }
      expect(
        mismatches,
        isEmpty,
        reason: 'Full frozen source token mismatches: $mismatches',
      );
    },
  );
  test('Changing or duplicating packet rows is rejected', () {
    final altered = full.cases
        .map((r) => Map<String, dynamic>.from(r))
        .toList();
    altered[0]['label'] = -1;
    expect(
      () => FullEvaluationPacket.validateRows(altered, study.gate),
      throwsFormatException,
    );
    altered[0] = altered[1];
    expect(
      () => FullEvaluationPacket.validateRows(altered, study.gate),
      throwsFormatException,
    );
  });
  test(
    'Full runner retains physical release and FP32 preflight restrictions',
    () {
      final p = StudyController()..packet = study;
      final c = FullEvaluationController(p)..packet = full;
      p.modelsReady = true;
      p.physicalDevice = true;
      p.variantPassed['fp32'] = true;
      expect(c.requiredRows, 3400);
      expect(
        c.canStart,
        false,
        reason: 'Debug mode cannot run a device evaluation',
      );
      c.dispose();
      p.dispose();
    },
  );
}
