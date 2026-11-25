import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:crisisq_device_study/core/frozen_tokenizer.dart';
import 'package:crisisq_device_study/core/frozen_gate.dart';
import 'package:crisisq_device_study/core/study_engine.dart';
import 'package:crisisq_device_study/core/csv_export.dart';

void main() {
  test('WordPiece matches all 256 actual frozen token references', () {
    final tokenizer = FrozenBertTokenizer.fromJson(
      File('test/fixtures/base_tokenizer_reference.json').readAsStringSync(),
    );
    final mismatch = <String>[];
    for (final line in const LineSplitter().convert(
      File('assets/study/benchmark_cases.jsonl').readAsStringSync(),
    )) {
      final c = jsonDecode(line) as Map;
      final e = tokenizer.encode(c['text'] as String, 128);
      if (!sameInts(e.inputIds, (c['input_ids'] as List).cast<int>()) ||
          !sameInts(
            e.attentionMask,
            (c['attention_mask'] as List).cast<int>(),
          )) {
        mismatch.add(c['source_id'] as String);
      }
    }
    expect(
      mismatch,
      isEmpty,
      reason: 'Tokenization mismatch IDs: ${mismatch.take(10).toList()}',
    );
  });
  test(
    'Frozen gate preserves conflict routing and actual packet decisions',
    () {
      final gate = FrozenGate.fromJson(
        File('assets/study/frozen_rules.json').readAsStringSync(),
      );
      expect(
        gate.route('I was not suicidal yesterday, but now I want to die.'),
        -1,
      );
      for (final line in const LineSplitter().convert(
        File('assets/study/benchmark_cases.jsonl').readAsStringSync(),
      )) {
        final c = jsonDecode(line) as Map;
        expect(gate.route(c['text'] as String), c['gate_v2']);
      }
    },
  );
  test('CSV quotes commas, quotes, nulls and line breaks', () {
    expect(csvField('a,"b"\nc'), '"a,""b""\nc"');
    expect(csvField(null), '');
    expect(
      csvRows(
        [
          {'a': true, 'b': null},
        ],
        columns: ['a', 'b'],
      ),
      'a,b\ntrue,\n',
    );
  });
  test('Probability parity rejects changed or invalid outputs', () {
    expect(positiveProbability([0, 0]), .5);
    expect(probabilitiesMatch(.5, .500001), isTrue);
    expect(probabilitiesMatch(.5, .6), isFalse);
    expect(
      () => positiveProbability([double.nan, 1]),
      throwsA(isA<ParityFailure>()),
    );
  });
}
