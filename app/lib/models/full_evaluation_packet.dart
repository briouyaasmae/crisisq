import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import '../core/frozen_gate.dart';
import 'study_packet.dart';

class FullEvaluationPacket {
  static const manifestSha =
      'b0f6f391c1277d826658bc11339ac47317d9766f0ae8c4b96be54fae77886f19';
  static const casesSha =
      'a24de8c2b432ddbe3880a579abc9106cde59d38b9f545eeae30f73ad955b1372';
  final Map<String, dynamic> manifest;
  final List<Map<String, dynamic>> cases;
  FullEvaluationPacket._(this.manifest, this.cases);
  static Future<FullEvaluationPacket> load(
    StudyPacket study, [
    AssetBundle? bundle,
  ]) async {
    final b = bundle ?? rootBundle;
    final m = await b.loadString('assets/study/full_evaluation_manifest.json');
    final c = await b.loadString('assets/study/full_evaluation_cases.jsonl');
    if (sha256.convert(utf8.encode(m)).toString() != manifestSha ||
        sha256.convert(utf8.encode(c)).toString() != casesSha) {
      throw const FormatException(
        'Full evaluation packet bytes differ. Restore the supplied frozen packet.',
      );
    }
    final manifest = Map<String, dynamic>.from(jsonDecode(m) as Map);
    if (manifest['source_external_hash'] !=
            'fe7125e21acb826cd46fa837388b55ff75dfe1b4177a68cdebf133d14ab1dfbf' ||
        manifest['tau'] != study.tau ||
        manifest['rules_sha256'] != study.rulesHash ||
        manifest['cases_sha256'] != casesSha) {
      throw const FormatException('Full evaluation protocol differs');
    }
    final hashes = Map<String, dynamic>.from(manifest['model_hashes'] as Map);
    if (hashes.length != study.expectedFiles.length ||
        !study.expectedFiles.entries.every((e) => hashes[e.key] == e.value)) {
      throw const FormatException('Full evaluation model hashes differ');
    }
    final cases = const LineSplitter()
        .convert(c)
        .where((l) => l.isNotEmpty)
        .map((l) => Map<String, dynamic>.from(jsonDecode(l) as Map))
        .toList();
    validateRows(cases, study.gate);
    return FullEvaluationPacket._(manifest, cases);
  }

  static void validateRows(List<Map<String, dynamic>> rows, FrozenGate gate) {
    final counts = <String, int>{};
    final ids = <String>{};
    for (final c in rows) {
      if (!ids.add(c['source_id'] as String) ||
          ![0, 1].contains(c['label']) ||
          (c['input_ids'] as List).length != 128 ||
          (c['attention_mask'] as List).length != 128 ||
          gate.route(c['text'] as String) != c['gate_v2']) {
        throw const FormatException(
          'Full evaluation row identity, tokens, labels or gate differ',
        );
      }
      final part = c['partition'] as String;
      counts[part] = (counts[part] ?? 0) + 1;
    }
    if (rows.length != 1700 ||
        counts.length != 3 ||
        counts['binary_test'] != 1000 ||
        counts['labeled_hard_negative'] != 300 ||
        counts['positive_probe'] != 400) {
      throw const FormatException(
        'Full evaluation requires the exact 1,700 frozen labeled cases',
      );
    }
  }
}
