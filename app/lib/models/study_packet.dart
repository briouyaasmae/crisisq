import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:crypto/crypto.dart';
import '../core/frozen_gate.dart';

class StudyPacket {
  final Map<String, dynamic> protocol, rules;
  final List<Map<String, dynamic>> cases;
  final String protocolHash, rulesHash, casesHash;
  final FrozenGate gate;
  StudyPacket._(
    this.protocol,
    this.rules,
    this.cases,
    this.protocolHash,
    this.rulesHash,
    this.casesHash,
  ) : gate = FrozenGate(rules);
  static Future<StudyPacket> load([AssetBundle? bundle]) async {
    final b = bundle ?? rootBundle;
    final p = await b.loadString('assets/study/device_protocol.json');
    final r = await b.loadString('assets/study/frozen_rules.json');
    final c = await b.loadString('assets/study/benchmark_cases.jsonl');
    final rows = const LineSplitter()
        .convert(c)
        .where((s) => s.trim().isNotEmpty)
        .map((s) => Map<String, dynamic>.from(jsonDecode(s) as Map))
        .toList();
    final packet = StudyPacket._(
      Map<String, dynamic>.from(jsonDecode(p) as Map),
      Map<String, dynamic>.from(jsonDecode(r) as Map),
      rows,
      sha256.convert(utf8.encode(p)).toString(),
      sha256.convert(utf8.encode(r)).toString(),
      sha256.convert(utf8.encode(c)).toString(),
    );
    if (packet.protocolHash !=
            'a053b4df62c213d9662f93d0ddd5abceb7f0d6a2d43e350fc30a6b5db8ed06b8' ||
        packet.rulesHash !=
            '661af3b209f5a166cef235c31462ce1daa90a6c92123dec6ef1247372ef04c29' ||
        packet.casesHash !=
            'cc335ea2fc4a2256a99c3b368c78c4f952f43265e3219c8bf4986fb8189690b6') {
      throw const FormatException('Frozen packet bytes changed');
    }
    if (rows.length != 256 ||
        packet.maxLength != 128 ||
        packet.repeats != 5 ||
        packet.warmups != 10 ||
        packet.evidenceHash !=
            '73eb4e3dbd50dd2a68b000a3e6bf58fd1f9f7a6d655a23b7069131c039d2ba7c') {
      throw const FormatException(
        'Benchmark packet differs from the completed frozen study',
      );
    }
    final ids = <String>{};
    for (final row in rows) {
      if (!ids.add(row['source_id'] as String)) {
        throw const FormatException('Duplicate benchmark input');
      }
      if ((row['input_ids'] as List).length != packet.maxLength ||
          (row['attention_mask'] as List).length != packet.maxLength ||
          packet.gate.route(row['text'] as String) != row['gate_v2']) {
        throw const FormatException(
          'Frozen token dimensions or gate parity failed',
        );
      }
    }
    for (final profile in packet.profiles) {
      if (rows.where((r) => r['partition'] == profile).length != 64) {
        throw const FormatException('Incomplete frozen profile');
      }
    }
    return packet;
  }

  String get evidenceHash => protocol['evidence_plan_hash'] as String;
  int get maxLength => (protocol['max_length'] as num).toInt();
  int get repeats => (protocol['repeats'] as num).toInt();
  int get warmups => (protocol['warmup_messages'] as num).toInt();
  int get orderSeed => (protocol['order_seed'] as num).toInt();
  double get tau => (protocol['tau'] as num).toDouble();
  List<String> get variants =>
      (protocol['models'] as Map).keys.cast<String>().toList()..sort();
  List<String> get profiles => (protocol['profiles'] as List).cast<String>();
  List<Map<String, dynamic>> profileCases(String p) =>
      cases.where((r) => r['partition'] == p).toList();
  double probability(Map<String, dynamic> row, String variant) =>
      (((row['reference'] as Map)[variant] as Map)['probability'] as num)
          .toDouble();
  Map<String, String> get expectedFiles {
    final result = <String, String>{};
    for (final value in (protocol['models'] as Map).values) {
      final files = (value as Map)['all_frozen_file_hashes'] as Map;
      for (final entry in files.entries) {
        final filename = (entry.key as String).split('/').last;
        final hash = entry.value as String;
        if (result.containsKey(filename) && result[filename] != hash) {
          throw const FormatException('Conflicting model manifest');
        }
        result[filename] = hash;
      }
    }
    return result;
  }
}
