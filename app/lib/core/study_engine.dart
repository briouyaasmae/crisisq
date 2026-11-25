import 'dart:math' as math;
import '../models/study_packet.dart';
import '../services/native_model.dart';
import 'frozen_tokenizer.dart';

class ParityFailure implements Exception {
  final String message;
  const ParityFailure(this.message);
  @override
  String toString() => message;
}

double positiveProbability(List<double> logits) {
  if (logits.length != 2 || logits.any((v) => !v.isFinite)) {
    throw const ParityFailure('Invalid model logits');
  }
  final maximum = math.max(logits[0], logits[1]);
  final a = math.exp(logits[0] - maximum), b = math.exp(logits[1] - maximum);
  return b / (a + b);
}

bool probabilitiesMatch(double actual, double expected) =>
    (actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs();
bool sameInts(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

Map<String, dynamic> checkTokenizer(
  StudyPacket packet,
  FrozenBertTokenizer tokenizer,
) {
  final failures = <Map<String, dynamic>>[];
  for (final c in packet.cases) {
    final e = tokenizer.encode(c['text'] as String, packet.maxLength);
    if (!sameInts(e.inputIds, (c['input_ids'] as List).cast<int>()) ||
        !sameInts(e.attentionMask, (c['attention_mask'] as List).cast<int>())) {
      failures.add({'source_id': c['source_id'], 'reason': 'token_ids_differ'});
    }
  }
  return {
    'checked': packet.cases.length,
    'passed': failures.isEmpty,
    'failures': failures,
  };
}

class StudyEngine {
  final StudyPacket packet;
  final FrozenBertTokenizer tokenizer;
  StudyEngine(this.packet, this.tokenizer);
  Future<Map<String, dynamic>> timedOne({
    required Map<String, dynamic> input,
    required String variant,
    required String mode,
    required bool fullPipeline,
    required LogitModel model,
    bool requireKaggleParity = true,
    double? deviceReferenceProbability,
  }) async {
    if (!requireKaggleParity && variant != 'q_quint8') {
      throw ArgumentError(
        'Only QUInt8 supports the explicitly exploratory reference policy.',
      );
    }
    final total = Stopwatch()..start();
    var gateUs = 0, tokenUs = 0, modelUs = 0;
    var route = -1;
    if (mode == 'hybrid') {
      final sw = Stopwatch()..start();
      route = packet.gate.route(input['text'] as String);
      sw.stop();
      gateUs = sw.elapsedMicroseconds;
    } else if (mode != 'neural') {
      throw ArgumentError('Unknown mode');
    }
    double? probability;
    int decision;
    if (route < 0) {
      List<int> ids, mask;
      if (fullPipeline) {
        final sw = Stopwatch()..start();
        final e = tokenizer.encode(input['text'] as String, packet.maxLength);
        sw.stop();
        tokenUs = sw.elapsedMicroseconds;
        ids = e.inputIds;
        mask = e.attentionMask;
      } else {
        ids = (input['input_ids'] as List).cast<int>();
        mask = (input['attention_mask'] as List).cast<int>();
      }
      final sw = Stopwatch()..start();
      final logits = await model.run(ids, mask);
      sw.stop();
      modelUs = sw.elapsedMicroseconds;
      probability = positiveProbability(logits);
      decision = probability >= packet.tau ? 1 : 0;
    } else {
      decision = route;
    }
    total.stop();
    final expectedP = packet.probability(input, variant);
    final expected = route < 0 ? (expectedP >= packet.tau ? 1 : 0) : route;
    final kaggleProbabilityMatch = probability == null
        ? null
        : probabilitiesMatch(probability, expectedP);
    final kaggleParity =
        decision == expected &&
        (probability == null || kaggleProbabilityMatch == true);
    if (requireKaggleParity && !kaggleParity) {
      throw ParityFailure(
        'Device output differs from the frozen reference for ${input['source_id']}. Export diagnostics; do not adjust the threshold.',
      );
    }
    if (!requireKaggleParity) {
      if (deviceReferenceProbability == null ||
          !deviceReferenceProbability.isFinite ||
          deviceReferenceProbability < 0 ||
          deviceReferenceProbability > 1) {
        throw const ParityFailure(
          'Portability study requires a complete finite device reference.',
        );
      }
      if (probability != null &&
          (!probabilitiesMatch(probability, deviceReferenceProbability) ||
              decision != (deviceReferenceProbability >= packet.tau ? 1 : 0))) {
        throw ParityFailure(
          'QUInt8 is not repeatable on this device for ${input['source_id']}. This task is stopped; export diagnostics.',
        );
      }
    }
    return {
      'profile': input['partition'],
      'source_id': input['source_id'],
      'variant': variant,
      'mode': mode,
      'tokenizer_mode': fullPipeline ? 'on_device' : 'pretokenized',
      'timing_scope': fullPipeline
          ? 'routing_tokenization_model_pipeline'
          : 'routing_and_model_only',
      'total_ms': total.elapsedMicroseconds / 1000,
      'gate_ms': gateUs / 1000,
      'tokenization_ms': tokenUs / 1000,
      'onnx_ms': route < 0 ? modelUs / 1000 : null,
      'invoked': route < 0,
      'probability': probability,
      'decision': decision,
      'parity_verified': kaggleParity,
      'validation_scope': requireKaggleParity
          ? 'frozen_kaggle_reference'
          : 'device_repeatability_with_recorded_kaggle_drift',
      'neural_reference_probability': expectedP,
      'kaggle_probability_match': kaggleProbabilityMatch,
      'device_reference_probability': deviceReferenceProbability,
      'device_repeatability_verified':
          !requireKaggleParity && probability != null ? true : null,
    };
  }
}
