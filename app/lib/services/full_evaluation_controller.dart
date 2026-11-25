import 'dart:convert';
import 'dart:io';
import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import 'package:crisisq_device_metrics/crisisq_device_metrics.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import '../core/csv_export.dart';
import '../core/study_engine.dart';
import '../models/full_evaluation_packet.dart';
import 'native_model.dart';
import 'study_controller.dart';

class _FullEvaluationPaused implements Exception {}

class FullEvaluationController extends ChangeNotifier {
  final StudyController parent;
  FullEvaluationController(this.parent) {
    parent.addListener(_parentChanged);
  }
  FullEvaluationPacket? packet;
  bool busy = false,
      cancelRequested = false,
      complete = false,
      _disposed = false;
  int savedRows = 0;
  String status = 'Loading the full frozen labeled packet.', error = '';
  Directory? runDirectory;
  static const variants = ['fp32', 'q_quint8'];
  static const chunkSize = 25;
  int get requiredRows => (packet?.cases.length ?? 1700) * 2;
  bool get canStart =>
      !busy &&
      !parent.busy &&
      packet != null &&
      parent.modelsReady &&
      parent.variantPassed['fp32'] == true &&
      parent.physicalDevice &&
      kReleaseMode;
  void _parentChanged() {
    if (!_disposed) notifyListeners();
  }

  void _say(String message) {
    status = message;
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    parent.removeListener(_parentChanged);
    super.dispose();
  }

  Future<void> initialize() async {
    try {
      if (parent.packet == null) {
        throw StateError(
          'Return to setup and load the original study packet first.',
        );
      }
      packet = await FullEvaluationPacket.load(parent.packet!);
      _say(
        'Exact packet ready: 1,000 binary, 300 hard-negative and 400 probe messages.',
      );
    } catch (e) {
      error = e.toString();
      _say('Full packet needs attention.');
    }
  }

  Future<Directory> _base() async {
    final root = await getApplicationDocumentsDirectory();
    final d = Directory('${root.path}/crisisq_full_evaluation_runs');
    await d.create(recursive: true);
    return d;
  }

  String _timestamp() => DateTime.now()
      .toUtc()
      .toIso8601String()
      .replaceAll(':', '')
      .replaceAll('.', '');
  Future<void> _json(File file, Object obj) async {
    await file.parent.create(recursive: true);
    final tmp = File('${file.path}.tmp');
    await tmp.writeAsString(jsonEncode(obj), flush: true);
    await tmp.rename(file.path);
  }

  Map<String, dynamic> _configuration() => {
    'protocol': 'full_labeled_device_evaluation_v1',
    'packet_manifest_sha256': FullEvaluationPacket.manifestSha,
    'cases_sha256': FullEvaluationPacket.casesSha,
    'source_external_hash': packet!.manifest['source_external_hash'],
    'source_evaluation_rows_hash':
        packet!.manifest['source_evaluation_rows_hash'],
    'model_hashes': parent.packet!.expectedFiles,
    'rules_sha256': parent.packet!.rulesHash,
    'device_id': parent.deviceLabel.trim(),
    'device': parent.device,
    'threads': parent.threads,
    'runtime': StudyController.runtimeLabel,
    'session_configs': NativeLogitModel.sessionConfigEntries,
    'provider': 'CPUExecutionProvider',
    'build_mode': kReleaseMode ? 'release' : 'debug',
    'tau': parent.packet!.tau,
    'models': variants,
    'case_count': packet!.cases.length,
    'required_neural_forward_passes': requiredRows,
    'derived_configuration_decisions': packet!.cases.length * 4,
    'chunk_size': chunkSize,
    'input_scope':
        'saved tokenizer IDs; classification only; no pipeline timing claim',
    'selection_scope':
        'same frozen source labels, groups and partitions; previously inspected data; no retuning',
  };
  Future<Directory> _folder(bool resume) async {
    final base = await _base();
    final config = _configuration();
    final fingerprint = sha256
        .convert(utf8.encode(jsonEncode(config)))
        .toString();
    final pointer = File('${base.path}/last_full_run.json');
    if (resume) {
      if (!await pointer.exists()) {
        throw StateError(
          'No saved full evaluation. Start a new full evaluation.',
        );
      }
      final p = jsonDecode(await pointer.readAsString()) as Map;
      final d = Directory(p['path'] as String);
      final m =
          jsonDecode(await File('${d.path}/run_manifest.json').readAsString())
              as Map;
      if (m['configuration_hash'] != fingerprint) {
        throw StateError(
          'Resume settings differ. Restore the same device label, thread count and device/packet settings.',
        );
      }
      return d;
    }
    final d = Directory('${base.path}/${_timestamp()}');
    await Directory('${d.path}/chunks').create(recursive: true);
    await _json(File('${d.path}/run_manifest.json'), {
      'run_id': d.uri.pathSegments.where((s) => s.isNotEmpty).last,
      'configuration_hash': fingerprint,
      'configuration': config,
      'started_utc': DateTime.now().toUtc().toIso8601String(),
    });
    await _json(File('${d.path}/preflight_report.json'), parent.parityReport);
    await _json(pointer, {'path': d.path});
    return d;
  }

  Future<Map<String, dynamic>> _thermal() async {
    try {
      return await CrisisqDeviceMetrics.snapshot();
    } catch (e) {
      return {'thermal_state': 'unavailable', 'error': e.toString()};
    }
  }

  bool _hot(Map<String, dynamic> m) => [
    'serious',
    'critical',
    'severe',
    'emergency',
    'shutdown',
  ].contains(m['thermal_state']);
  void cancel() {
    cancelRequested = true;
    _say('Pausing. Completed 25-message chunks are preserved.');
  }

  Future<void> run({bool resume = false}) async {
    if (!canStart) {
      error =
          'Use a physical release device. Verify model files and pass the 256-case FP32 reference checks on the setup screen first.';
      _say('Setup checks required.');
      return;
    }
    busy = true;
    complete = false;
    cancelRequested = false;
    savedRows = 0;
    error = '';
    notifyListeners();
    try {
      await WakelockPlus.enable();
      runDirectory = await _folder(resume);
      for (final variant in variants) {
        LogitModel? model;
        try {
          for (
            var start = 0;
            start < packet!.cases.length;
            start += chunkSize
          ) {
            final end = (start + chunkSize)
                .clamp(0, packet!.cases.length)
                .toInt();
            final target = File(
              '${runDirectory!.path}/chunks/${variant}__${start.toString().padLeft(6, '0')}.json',
            );
            if (await target.exists()) {
              final old = jsonDecode(await target.readAsString()) as Map;
              final oldRows = old['rows'] as List;
              if (old['variant'] != variant ||
                  old['start'] != start ||
                  oldRows.length != end - start ||
                  !List.generate(
                    end - start,
                    (i) =>
                        oldRows[i]['source_id'] ==
                        packet!.cases[start + i]['source_id'],
                  ).every((v) => v)) {
                throw StateError(
                  'Saved chunk identity differs. Preserve the export and inspect it.',
                );
              }
              savedRows += oldRows.length;
              continue;
            }
            if (cancelRequested) throw _FullEvaluationPaused();
            _say(
              'Cooling pause · $savedRows/$requiredRows model outputs saved',
            );
            await Future<void>.delayed(const Duration(seconds: 5));
            if (cancelRequested) throw _FullEvaluationPaused();
            final before = await _thermal();
            if (_hot(before)) {
              throw StateError(
                'OS thermal state is ${before['thermal_state']}. Let the device cool, then resume.',
              );
            }
            if (model == null) {
              _say('Opening $variant for the full classification evaluation');
              await SchedulerBinding.instance.endOfFrame;
              model = await NativeLogitModel.open(
                parent.store!.file('$variant.onnx').path,
                parent.threads,
              );
            }
            final rows = <Map<String, dynamic>>[];
            for (var i = start; i < end; i++) {
              if (cancelRequested) throw _FullEvaluationPaused();
              final c = packet!.cases[i];
              final probability = positiveProbability(
                await model.run(
                  (c['input_ids'] as List).cast<int>(),
                  (c['attention_mask'] as List).cast<int>(),
                ),
              );
              if (cancelRequested) throw _FullEvaluationPaused();
              final reference =
                  (((c['reference'] as Map)[variant] as Map)['probability']
                          as num)
                      .toDouble();
              final neural = probability >= parent.packet!.tau ? 1 : 0;
              final gate = c['gate_v2'] as int;
              rows.add({
                'variant': variant,
                'source_id': c['source_id'],
                'text_hash': c['text_hash'],
                'group_id': c['group_id'],
                'partition': c['partition'],
                'label': c['label'],
                'gate_v2': gate,
                'probability': probability,
                'prediction_neural': neural,
                'prediction_hybrid': gate < 0 ? neural : gate,
                'reference_probability': reference,
                'kaggle_probability_match': probabilitiesMatch(
                  probability,
                  reference,
                ),
                'kaggle_decision_match':
                    neural == (reference >= parent.packet!.tau ? 1 : 0),
                'scored_utc': DateTime.now().toUtc().toIso8601String(),
              });
              if ((i - start) % 5 == 0) {
                _say(
                  '$variant · message ${i + 1}/${packet!.cases.length} · $savedRows/$requiredRows saved',
                );
                await SchedulerBinding.instance.endOfFrame;
              }
            }
            final after = await _thermal();
            await _json(target, {
              'variant': variant,
              'start': start,
              'end': end,
              'before': before,
              'after': after,
              'rows': rows,
            });
            savedRows += rows.length;
            _say('$savedRows/$requiredRows model outputs saved');
            if (_hot(after)) {
              throw StateError(
                'OS thermal state is ${after['thermal_state']}. This completed chunk is saved. Cool the device and resume.',
              );
            }
          }
        } finally {
          await model?.close();
        }
      }
      complete = true;
      await _json(File('${runDirectory!.path}/completion.json'), {
        'complete': true,
        'model_outputs': savedRows,
        'unique_messages': packet!.cases.length,
        'derived_configuration_decisions': packet!.cases.length * 4,
        'finished_utc': DateTime.now().toUtc().toIso8601String(),
      });
      _say(
        'Full evaluation complete: 3,400 model outputs and four configurations. Export results.',
      );
    } on _FullEvaluationPaused {
      _say(
        'Paused. Completed chunks remain saved. Resume with the same settings.',
      );
    } catch (e) {
      error = e.toString();
      _say('Full evaluation paused. Completed chunks remain saved.');
    } finally {
      busy = false;
      cancelRequested = false;
      try {
        await WakelockPlus.disable();
      } catch (_) {}
      if (!_disposed) notifyListeners();
    }
  }

  Future<String> makeExport() async {
    if (runDirectory == null) {
      throw StateError('Start or resume the full evaluation first.');
    }
    final archive = Archive();
    void add(String name, String text) {
      final bytes = utf8.encode(text);
      archive.addFile(ArchiveFile(name, bytes.length, bytes));
    }

    for (final name in ['run_manifest.json', 'preflight_report.json']) {
      add(name, await File('${runDirectory!.path}/$name').readAsString());
    }
    add(
      'packet_manifest.json',
      const JsonEncoder.withIndent('  ').convert(packet!.manifest),
    );
    final rows = <Map<String, dynamic>>[], thermal = <Map<String, dynamic>>[];
    for (final variant in variants) {
      for (var start = 0; start < packet!.cases.length; start += chunkSize) {
        final file = File(
          '${runDirectory!.path}/chunks/${variant}__${start.toString().padLeft(6, '0')}.json',
        );
        if (!await file.exists()) continue;
        final obj = jsonDecode(await file.readAsString()) as Map;
        rows.addAll(
          (obj['rows'] as List).map((r) => Map<String, dynamic>.from(r as Map)),
        );
        thermal.add({
          'variant': variant,
          'start': start,
          'end': obj['end'],
          'before': obj['before'],
          'after': obj['after'],
        });
      }
    }
    final done = complete && rows.length == requiredRows;
    if (rows.isNotEmpty) {
      add(
        done
            ? 'full_device_predictions.csv'
            : 'full_device_predictions_INCOMPLETE.csv',
        csvRows(rows, columns: rows.first.keys.toList()),
      );
    }
    add(
      'completion.json',
      jsonEncode({
        'complete': done,
        'model_outputs': rows.length,
        'required_model_outputs': requiredRows,
        'unique_messages_required': packet!.cases.length,
        'derived_configuration_decisions': rows.length * 2,
      }),
    );
    add('thermal_snapshots.json', jsonEncode(thermal));
    add(
      'READ_ME.txt',
      'Full frozen-source classification evaluation. One neural pass per message per model; neural and hybrid decisions derived from that probability and the frozen rule gate. No repeated timing rows or new independent data. QUInt8 Kaggle drift is recorded, not treated as equivalence. Source labels are nonclinical. Raw messages and model weights are excluded. Analyze with analyze_full_device_evaluation.py.\n',
    );
    final base = await _base();
    final file = File(
      '${base.path}/crisisq_full_device_evaluation_${_timestamp()}.zip',
    );
    await file.writeAsBytes(ZipEncoder().encode(archive), flush: true);
    return file.path;
  }
}
