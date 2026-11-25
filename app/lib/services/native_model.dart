import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

abstract class LogitModel {
  Future<List<double>> run(List<int> ids, List<int> mask);
  Future<void> close();
}

class NativeLogitModel implements LogitModel {
  // Explicitly recorded in every reference check and exported run.
  static const sessionConfigEntries = <String, String>{};
  final OrtSession session;
  NativeLogitModel._(this.session);
  static Future<NativeLogitModel> open(String path, int threads) async {
    final runtime = OnnxRuntime();
    final session = await runtime.createSession(
      path,
      options: OrtSessionOptions(
        intraOpNumThreads: threads,
        interOpNumThreads: 1,
        providers: [OrtProvider.CPU],
        useArena: true,
        sessionConfigs: sessionConfigEntries,
      ),
    );
    if (session.inputNames.toSet().difference({
          'input_ids',
          'attention_mask',
        }).isNotEmpty ||
        !session.inputNames.contains('input_ids') ||
        !session.inputNames.contains('attention_mask') ||
        !session.outputNames.contains('logits')) {
      await session.close();
      throw StateError('Unexpected frozen ONNX input/output names');
    }
    return NativeLogitModel._(session);
  }

  @override
  Future<List<double>> run(List<int> ids, List<int> mask) async {
    final inputs = <String, OrtValue>{};
    Map<String, OrtValue> outputs = {};
    try {
      inputs['input_ids'] = await OrtValue.fromList(Int64List.fromList(ids), [
        1,
        ids.length,
      ]);
      inputs['attention_mask'] = await OrtValue.fromList(
        Int64List.fromList(mask),
        [1, mask.length],
      );
      outputs = await session.run(inputs);
      final logits = await outputs['logits']!.asFlattenedList();
      if (logits.length != 2) {
        throw StateError('Frozen classifier must output two logits');
      }
      return logits.map((n) => (n as num).toDouble()).toList();
    } finally {
      for (final tensor in outputs.values) {
        await tensor.dispose();
      }
      for (final tensor in inputs.values) {
        await tensor.dispose();
      }
    }
  }

  @override
  Future<void> close() => session.close();
}
