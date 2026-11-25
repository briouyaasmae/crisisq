import 'dart:convert';

class GateTrace {
  final int decision;
  final String reason;
  final List<String> affirmed, negated, safe;
  const GateTrace(
    this.decision,
    this.reason,
    this.affirmed,
    this.negated,
    this.safe,
  );
}

class FrozenGate {
  final Map<String, dynamic> spec;
  late final List<List<dynamic>> crisis, safe;
  late final Set<String> negators;
  late final RegExp contrast, risk;
  FrozenGate(this.spec) {
    crisis = (spec['crisis'] as List)
        .map((x) => [x[0] as String, RegExp(x[2] as String)])
        .toList();
    safe = (spec['safe'] as List)
        .map((x) => [x[0] as String, RegExp(x[2] as String)])
        .toList();
    negators = (spec['negators'] as List).cast<String>().toSet();
    contrast = RegExp(spec['contrast'] as String);
    risk = RegExp(spec['risk'] as String);
  }
  factory FrozenGate.fromJson(String s) =>
      FrozenGate(jsonDecode(s) as Map<String, dynamic>);
  String normalize(String text) {
    var s = text.toLowerCase();
    for (final ch in ["'", '’', '`', '‘']) {
      s = s.replaceAll(ch, '');
    }
    return s
        .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  GateTrace trace(String text) {
    final t = normalize(text);
    final affirmed = <String>[], negated = <String>[], hits = <String>[];
    final spans = <List<int>>[];
    for (final rule in crisis) {
      for (final m in (rule[1] as RegExp).allMatches(t)) {
        final prior = t.substring(0, m.start).trim();
        final words = prior.isEmpty ? <String>[] : prior.split(RegExp(r'\s+'));
        final start = words.length > 4 ? words.length - 4 : 0;
        final isNegated = words.sublist(start).any(negators.contains);
        (isNegated ? negated : affirmed).add(rule[0] as String);
      }
    }
    for (final rule in safe) {
      for (final m in (rule[1] as RegExp).allMatches(t)) {
        hits.add(rule[0] as String);
        spans.add([m.start, m.end]);
      }
    }
    int decision;
    String reason;
    if (affirmed.isNotEmpty && hits.isNotEmpty) {
      decision = -1;
      reason = 'conflict_to_stage2';
    } else if (affirmed.isNotEmpty) {
      decision = 1;
      reason = 'crisis';
    } else if (negated.isNotEmpty) {
      decision = -1;
      reason = 'negated_crisis_to_stage2';
    } else if (hits.isNotEmpty) {
      var residual = t;
      spans.sort(
        (a, b) => a[0] == b[0] ? b[1].compareTo(a[1]) : b[0].compareTo(a[0]),
      );
      for (final span in spans) {
        final a = span[0] > residual.length ? residual.length : span[0];
        final b = span[1] > residual.length ? residual.length : span[1];
        residual = '${residual.substring(0, a)} ${residual.substring(b)}';
      }
      if (contrast.hasMatch(t) || risk.hasMatch(residual)) {
        decision = -1;
        reason = 'safe_with_residual_risk_to_stage2';
      } else {
        decision = 0;
        reason = 'safe_context';
      }
    } else {
      decision = -1;
      reason = 'uncertain';
    }
    return GateTrace(
      decision,
      reason,
      affirmed.toSet().toList()..sort(),
      negated.toSet().toList()..sort(),
      hits.toSet().toList()..sort(),
    );
  }

  int route(String text) => trace(text).decision;
}
