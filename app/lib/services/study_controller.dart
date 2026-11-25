import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import 'package:crisisq_device_metrics/crisisq_device_metrics.dart';
import '../models/study_packet.dart';
import '../core/frozen_tokenizer.dart';
import '../core/study_engine.dart';
import '../core/csv_export.dart';
import 'model_store.dart';
import 'native_model.dart';

class CancelledStudy implements Exception {}

class StudyController extends ChangeNotifier {
  StudyPacket? packet;
  ModelStore? store;
  FrozenBertTokenizer? tokenizer;
  StudyEngine? engine;
  bool busy = false,
      modelsReady = false,
      tokenizerPassed = false,
      parityPassed = false;
  bool physicalDevice = false, fullPipeline = true, cancelRequested = false;
  bool fp32Only = true;
  bool allowQuint8Drift = false;
  bool get quint8Portability => !fp32Only && allowQuint8Drift;
  Map<String, double> get quint8DeviceReferences {
    final values = <String, double>{};
    for (final row in (parityReport['models'] as List? ?? const [])) {
      if (row['variant'] == 'q_quint8' && row['actual_probability'] is num) {
        final value = (row['actual_probability'] as num).toDouble();
        if (value.isFinite && value >= 0 && value <= 1) {
          values[row['source_id'] as String] = value;
        }
      }
    }
    return values;
  }

  bool get quint8DeviceReferencesReady {
    if (packet == null) return false;
    final values = quint8DeviceReferences;
    return values.length == packet!.cases.length &&
        packet!.cases.every((c) => values.containsKey(c['source_id']));
  }

  String get quint8DiagnosticSummary {
    final rows = (parityReport['models'] as List? ?? const [])
        .where((r) => r['variant'] == 'q_quint8')
        .toList();
    if (rows.isEmpty) return 'QUInt8 reference checks pending.';
    final failures = rows.where((r) => r['passed'] != true).length;
    final tau = packet?.tau ?? 0.6455836149936347;
    final flips = rows
        .where(
          (r) =>
              ((r['actual_probability'] as num) >= tau) !=
              ((r['expected_probability'] as num) >= tau),
        )
        .length;
    return 'QUInt8: ${rows.length}/256 outputs captured; $failures Kaggle mismatches; $flips changed decisions. These remain recorded in the export.';
  }

  bool get selectedModelsReady => selectedVariants.every(
    (v) =>
        variantPassed[v] == true ||
        (v == 'q_quint8' && quint8Portability && quint8DeviceReferencesReady),
  );

  final Map<String, bool> variantPassed = {};
  List<String> get selectedVariants =>
      fp32Only ? ['fp32'] : (packet?.variants ?? ['fp32', 'q_quint8']);
  bool get selectedReferencesPassed =>
      selectedVariants.every((v) => variantPassed[v] == true);
  int get studyTaskCount =>
      (packet?.profiles.length ?? 4) * selectedVariants.length * 2;
  int get studyMeasurements =>
      (packet?.cases.length ?? 256) *
      selectedVariants.length *
      2 *
      (packet?.repeats ?? 5);
  int threads = 4, completedTasks = 0;
  double progress = 0;
  String status = 'Load and verify the benchmark assets.',
      error = '',
      deviceLabel = '';
  final Map<String, dynamic> device = {};
  Map<String, dynamic> parityReport = {};
  Directory? runDirectory;
  String? exportPath;
  bool complete = false;
  static const runtimeLabel =
      'flutter_onnxruntime 1.9.0 / ORT 1.28.0 (pinned Android/iOS dependency)';
  bool get canBenchmark =>
      !busy &&
      modelsReady &&
      selectedModelsReady &&
      (!fullPipeline || tokenizerPassed) &&
      physicalDevice &&
      kReleaseMode;
  void configure({
    int? threadCount,
    bool? useTokenizer,
    bool? onlyFp32,
    bool? permitQuint8Drift,
  }) {
    if (busy) return;
    if (threadCount != null && threadCount != threads) {
      threads = threadCount;
      parityPassed = false;
      variantPassed.clear();
      parityReport = parityReport.containsKey('tokenizer')
          ? {'tokenizer': parityReport['tokenizer']}
          : {};
    }
    if (useTokenizer != null) fullPipeline = useTokenizer;
    if (onlyFp32 != null) fp32Only = onlyFp32;
    if (permitQuint8Drift != null) allowQuint8Drift = permitQuint8Drift;
    runDirectory = null;
    complete = false;
    exportPath = null;
    notifyListeners();
  }

  void _say(String text) {
    status = text;
    notifyListeners();
  }

  Future<void> initialize() async {
    try {
      packet = await StudyPacket.load();
      store = ModelStore(packet!);
      final info = DeviceInfoPlugin();
      if (Platform.isAndroid) {
        final a = await info.androidInfo;
        device.addAll({
          'model': '${a.brand} ${a.model}',
          'os': 'Android ${a.version.release} / SDK ${a.version.sdkInt}',
          'physical': a.isPhysicalDevice,
        });
      } else if (Platform.isIOS) {
        final i = await info.iosInfo;
        device.addAll({
          'model': i.utsname.machine,
          'os': 'iOS ${i.systemVersion}',
          'physical': i.isPhysicalDevice,
        });
      } else {
        device.addAll({
          'model': Platform.operatingSystem,
          'os': Platform.operatingSystemVersion,
          'physical': false,
        });
      }
      physicalDevice = device['physical'] == true;
      deviceLabel = device['model'] as String;
      _say('Frozen packet loaded: 256 messages, four profiles.');
    } catch (e) {
      error = e.toString();
      _say('Setup needs attention.');
    }
  }

  Future<Map<String, dynamic>> _metrics() async {
    try {
      return await CrisisqDeviceMetrics.snapshot();
    } catch (e) {
      return {
        'thermal_state': 'unavailable',
        'memory_metric': 'unavailable',
        'memory_value': null,
        'error': e.toString(),
      };
    }
  }

  bool _tooHot(Map<String, dynamic> m) => [
    'serious',
    'critical',
    'severe',
    'emergency',
    'shutdown',
  ].contains(m['thermal_state']);
  Future<void> prepare() async {
    if (busy || packet == null) return;
    busy = true;
    error = '';
    parityPassed = false;
    variantPassed.clear();
    notifyListeners();
    try {
      await store!.prepare(_say);
      tokenizer = FrozenBertTokenizer.fromJson(
        await store!.file('tokenizer.json').readAsString(),
      );
      engine = StudyEngine(packet!, tokenizer!);
      final check = checkTokenizer(packet!, tokenizer!);
      tokenizerPassed = check['passed'] == true;
      parityReport = {'tokenizer': check};
      modelsReady = true;
      _say(
        tokenizerPassed
            ? 'Five frozen files verified. Tokenization matches all 256 references.'
            : 'Files verified. Tokenizer mismatch: only pretokenized validation is available.',
      );
    } catch (e) {
      error = e.toString();
      modelsReady = false;
      _say('Model setup needs attention.');
    } finally {
      busy = false;
      notifyListeners();
    }
  }

  Future<void> _json(File file, Object obj) async {
    await file.parent.create(recursive: true);
    final tmp = File('${file.path}.tmp');
    await tmp.writeAsString(
      const JsonEncoder.withIndent('  ').convert(obj),
      flush: true,
    );
    await tmp.rename(file.path);
  }

  Future<Directory> _base() async {
    final d = await getApplicationDocumentsDirectory();
    final base = Directory('${d.path}/crisisq_runs');
    await base.create(recursive: true);
    return base;
  }

  String _timestamp() => DateTime.now()
      .toUtc()
      .toIso8601String()
      .replaceAll(':', '')
      .replaceAll('.', '');
  Future<void> checkReferences() async {
    if (busy || !modelsReady) return;
    busy = true;
    parityPassed = false;
    variantPassed.clear();
    // A completed new check must not retain a prior cancellation/error record.
    parityReport = parityReport.containsKey('tokenizer')
        ? {'tokenizer': parityReport['tokenizer']}
        : {};
    error = '';
    progress = 0;
    notifyListeners();
    final records = <Map<String, dynamic>>[];
    var failures = 0;
    try {
      for (var v = 0; v < packet!.variants.length; v++) {
        final variant = packet!.variants[v];
        LogitModel? model;
        try {
          _say('Reference check: $variant');
          model = await NativeLogitModel.open(
            store!.file('$variant.onnx').path,
            threads,
          );
          for (var i = 0; i < packet!.cases.length; i++) {
            if (cancelRequested) throw CancelledStudy();
            final c = packet!.cases[i];
            final actual = positiveProbability(
              await model.run(
                (c['input_ids'] as List).cast<int>(),
                (c['attention_mask'] as List).cast<int>(),
              ),
            );
            final expected = packet!.probability(c, variant);
            final passed =
                probabilitiesMatch(actual, expected) &&
                (actual >= packet!.tau) == (expected >= packet!.tau);
            if (!passed) failures++;
            records.add({
              'variant': variant,
              'source_id': c['source_id'],
              'expected_probability': expected,
              'actual_probability': actual,
              'absolute_error': (actual - expected).abs(),
              'passed': passed,
            });
            progress =
                (v * packet!.cases.length + i + 1) /
                (packet!.variants.length * packet!.cases.length);
            if (i % 8 == 0) {
              notifyListeners();
              await SchedulerBinding.instance.endOfFrame;
            }
          }
        } finally {
          await model?.close();
        }
        final checked = records.where((r) => r['variant'] == variant).toList();
        variantPassed[variant] =
            checked.length == packet!.cases.length &&
            checked.every((r) => r['passed'] == true);
      }
      parityPassed = failures == 0;
      parityReport.addAll({
        'models': records,
        'failures': failures,
        'passed': parityPassed,
        'threads': threads,
        'runtime': runtimeLabel,
      });
      _say(
        parityPassed
            ? 'Both models match every frozen reference. Ready for release-device timing.'
            : variantPassed['fp32'] == true
            ? '$failures QUInt8 Kaggle mismatches. FP32 passed. FP32-only is available; an explicit QUInt8 portability study retains these mismatches and checks device repeatability.'
            : '$failures reference mismatches. Selected models must pass before timing.',
      );
    } catch (e) {
      error = e.toString();
      parityReport.addAll({'models': records, 'passed': false, 'error': error});
      _say('Reference validation stopped.');
    } finally {
      parityReport['session_configs'] = NativeLogitModel.sessionConfigEntries;
      parityReport['threads'] = threads;
      parityReport['runtime'] = runtimeLabel;
      parityReport['variant_passed'] = Map<String, bool>.from(variantPassed);
      cancelRequested = false;
      busy = false;
      final base = await _base();
      await _json(File('${base.path}/latest_parity_report.json'), parityReport);
      notifyListeners();
    }
  }

  Map<String, dynamic> _configuration() => {
    'evidence_plan_hash': packet!.evidenceHash,
    'packet_sha256': packet!.protocolHash,
    'rules_sha256': packet!.rulesHash,
    'cases_sha256': packet!.casesHash,
    'model_hashes': packet!.expectedFiles,
    'device_id': deviceLabel.trim(),
    'device': device,
    'threads': threads,
    'full_pipeline': fullPipeline,
    'selected_models': selectedVariants,
    'study_scope': fp32Only
        ? 'fp32_only'
        : quint8Portability
        ? 'fp32_and_quint8_portability'
        : 'fp32_and_int8',
    'allow_quint8_cross_runtime_drift': quint8Portability,
    'measurement_protocol': 'quint8_portability_v1',
    'session_configs': NativeLogitModel.sessionConfigEntries,
    'quint8_reference_policy': quint8Portability
        ? 'record_kaggle_drift_require_device_repeatability'
        : 'strict_kaggle_equivalence',
    'required_tasks': studyTaskCount,
    'required_measurements': studyMeasurements,
    'tau': packet!.tau,
    'runtime': runtimeLabel,
    'provider': 'CPUExecutionProvider',
    'repeats': packet!.repeats,
    'warmups': packet!.warmups,
    'build_mode': kReleaseMode
        ? 'release'
        : kProfileMode
        ? 'profile'
        : 'debug',
    'app_version': '1.1.0',
  };
  Future<Directory> _runFolder(bool resume) async {
    final base = await _base();
    final config = _configuration();
    final hash = sha256.convert(utf8.encode(jsonEncode(config))).toString();
    final pointer = File('${base.path}/last_run.json');
    if (resume && await pointer.exists()) {
      final p = jsonDecode(await pointer.readAsString()) as Map;
      final dir = Directory(p['path'] as String);
      final manifest = File('${dir.path}/run_manifest.json');
      if (await manifest.exists()) {
        final m = jsonDecode(await manifest.readAsString()) as Map;
        if (m['configuration_hash'] != hash) {
          throw StateError(
            'Resume settings differ. Restore the same device label, threads and tokenization mode, or start a new run.',
          );
        }
        return dir;
      }
    }
    if (resume) throw StateError('No matching previous run is available.');
    final dir = Directory('${base.path}/${_timestamp()}');
    await Directory('${dir.path}/tasks').create(recursive: true);
    await _json(File('${dir.path}/run_manifest.json'), {
      'run_id': dir.uri.pathSegments.where((s) => s.isNotEmpty).last,
      'configuration_hash': hash,
      'configuration': config,
      'started_utc': DateTime.now().toUtc().toIso8601String(),
      'study_scope':
          'matched frozen-model timing; no new model selection or clinical validation',
      'memory_scope':
          'current whole-app snapshots, never peak or model-only; Android PSS and iOS RSS differ',
    });
    await _json(File('${dir.path}/parity_report.json'), parityReport);
    await _json(pointer, {'path': dir.path});
    return dir;
  }

  List<Map<String, String>> _tasks([List<String>? variants]) {
    final tasks = <Map<String, String>>[];
    for (final profile in packet!.profiles) {
      for (final variant in variants ?? selectedVariants) {
        for (final mode in ['neural', 'hybrid']) {
          tasks.add({
            'profile': profile,
            'variant': variant,
            'mode': mode,
            'id': '${profile}__${variant}__$mode',
          });
        }
      }
    }
    tasks.shuffle(Random(packet!.orderSeed));
    return tasks;
  }

  void cancel() {
    cancelRequested = true;
    _say('Pausing after the current message.');
  }

  Future<void> runStudy({bool resume = false}) async {
    if (!canBenchmark) {
      error =
          'Use a physical release device. FP32 must pass. QUInt8 needs strict parity or explicit portability mode with complete device references.';
      notifyListeners();
      return;
    }
    busy = true;
    cancelRequested = false;
    error = '';
    complete = false;
    progress = 0;
    completedTasks = 0;
    notifyListeners();
    try {
      await WakelockPlus.enable();
      runDirectory = await _runFolder(resume);
      final tasks = _tasks();
      for (final task in tasks) {
        final target = File('${runDirectory!.path}/tasks/${task['id']}.json');
        if (await target.exists()) {
          final saved = jsonDecode(await target.readAsString()) as Map;
          if (saved['task_id'] != task['id'] ||
              (saved['records'] as List).length != 64 * packet!.repeats) {
            throw StateError(
              'Incomplete cached task. Preserve the run and inspect it.',
            );
          }
          completedTasks++;
          progress = completedTasks / tasks.length;
          continue;
        }
        if (cancelRequested) throw CancelledStudy();
        _say('Cooling pause before task ${completedTasks + 1}/${tasks.length}');
        await Future<void>.delayed(const Duration(seconds: 5));
        final before = await _metrics();
        if (_tooHot(before)) {
          throw StateError(
            'OS thermal state is ${before['thermal_state']}. Let the phone cool, then resume.',
          );
        }
        final variant = task['variant']!,
            mode = task['mode']!,
            profile = task['profile']!;
        final started = DateTime.now().toUtc().toIso8601String();
        final snapshots = <Map<String, dynamic>>[
          {'stage': 'before_session', ...before},
        ];
        final load = Stopwatch();
        LogitModel? model;
        final records = <Map<String, dynamic>>[];
        try {
          _say('Loading $variant · $mode · $profile');
          await SchedulerBinding.instance.endOfFrame;
          load.start();
          model = await NativeLogitModel.open(
            store!.file('$variant.onnx').path,
            threads,
          );
          load.stop();
          snapshots.add({'stage': 'after_session', ...await _metrics()});
          final cases = packet!.profileCases(profile);
          final deviceReferences = quint8Portability && variant == 'q_quint8'
              ? quint8DeviceReferences
              : <String, double>{};
          for (var i = 0; i < packet!.warmups; i++) {
            if (cancelRequested) throw CancelledStudy();
            await engine!.timedOne(
              input: cases[i % cases.length],
              variant: variant,
              mode: mode,
              fullPipeline: fullPipeline,
              model: model,
              requireKaggleParity:
                  !(quint8Portability && variant == 'q_quint8'),
              deviceReferenceProbability:
                  deviceReferences[cases[i % cases.length]['source_id']],
            );
          }
          final warm = await _metrics();
          snapshots.add({'stage': 'after_warmup', ...warm});
          if (_tooHot(warm)) {
            throw StateError(
              'OS thermal state became ${warm['thermal_state']}; cool the phone and resume.',
            );
          }
          final context = {
            'device_id': deviceLabel.trim(),
            'os': device['os'],
            'runtime': runtimeLabel,
            'provider': 'CPUExecutionProvider',
            'threads': threads,
            'thermal_state': warm['thermal_state'] ?? 'unavailable',
            'evidence_plan_hash': packet!.evidenceHash,
            'run_id': runDirectory!.uri.pathSegments
                .where((s) => s.isNotEmpty)
                .last,
            'task_id': task['id'],
            'task_started_utc': started,
            'build_mode': 'release',
          };
          for (var repeat = 0; repeat < packet!.repeats; repeat++) {
            final order = List<int>.generate(cases.length, (i) => i)
              ..shuffle(Random(packet!.orderSeed + repeat));
            for (var seq = 0; seq < order.length; seq++) {
              if (cancelRequested) throw CancelledStudy();
              final row = await engine!.timedOne(
                input: cases[order[seq]],
                variant: variant,
                mode: mode,
                fullPipeline: fullPipeline,
                model: model,
                requireKaggleParity:
                    !(quint8Portability && variant == 'q_quint8'),
                deviceReferenceProbability:
                    deviceReferences[cases[order[seq]]['source_id']],
              );
              if (cancelRequested) throw CancelledStudy();
              records.add({
                ...context,
                ...row,
                'repeat': repeat,
                'sequence': seq,
              });
              if (seq % 8 == 0) {
                progress =
                    (completedTasks +
                        (repeat + seq / cases.length) / packet!.repeats) /
                    tasks.length;
                _say(
                  'Task ${completedTasks + 1}/${tasks.length} · repeat ${repeat + 1}/5 · ${seq + 1}/64',
                );
                await SchedulerBinding.instance.endOfFrame;
              }
            }
            await _json(File('${target.path}.partial'), {
              'task_id': task['id'],
              'completed': false,
              'records': records,
            });
          }
          final afterMessages = await _metrics();
          snapshots.add({'stage': 'after_messages', ...afterMessages});
          if (_tooHot(afterMessages)) {
            throw StateError(
              'OS thermal state is ${afterMessages['thermal_state']} after measurement. Cool the phone and resume; this task will restart.',
            );
          }
          await _json(target, {
            'task_id': task['id'],
            'completed': true,
            'variant': variant,
            'mode': mode,
            'profile': profile,
            'session_init_ms': load.elapsedMicroseconds / 1000,
            'snapshots': snapshots,
            'records': records,
          });
          final partial = File('${target.path}.partial');
          if (await partial.exists()) await partial.delete();
          completedTasks++;
          progress = completedTasks / tasks.length;
        } finally {
          await model?.close();
        }
      }
      complete = true;
      progress = 1;
      await _json(File('${runDirectory!.path}/completion.json'), {
        'complete': true,
        'tasks': tasks.length,
        'measurements': studyMeasurements,
        'selected_models': selectedVariants,
        'evidence_plan_hash': packet!.evidenceHash,
        'finished_utc': DateTime.now().toUtc().toIso8601String(),
      });
      _say(
        'All ${tasks.length} tasks complete (${selectedVariants.join(', ')}). Export the measured results.',
      );
    } on CancelledStudy {
      _say(
        'Paused. Completed tasks are preserved; an interrupted task restarts after warmup.',
      );
    } catch (e) {
      error = e.toString();
      _say('Study paused. Completed tasks remain saved.');
    } finally {
      cancelRequested = false;
      busy = false;
      try {
        await WakelockPlus.disable();
      } catch (_) {}
      notifyListeners();
    }
  }

  Future<String> makeExport() async {
    final base = await _base();
    final archive = Archive();
    void addText(String name, String text) {
      final bytes = utf8.encode(text);
      archive.addFile(ArchiveFile(name, bytes.length, bytes));
    }

    final savedParity = runDirectory == null
        ? null
        : File('${runDirectory!.path}/parity_report.json');
    addText(
      'parity_report.json',
      savedParity != null && await savedParity.exists()
          ? await savedParity.readAsString()
          : const JsonEncoder.withIndent('  ').convert(parityReport),
    );
    addText(
      'latest_diagnostics.json',
      const JsonEncoder.withIndent('  ').convert(parityReport),
    );
    final savedManifest = runDirectory == null
        ? null
        : File('${runDirectory!.path}/run_manifest.json');
    final exportedContext =
        savedManifest != null && await savedManifest.exists()
        ? (jsonDecode(await savedManifest.readAsString())
              as Map)['configuration']
        : _configuration();
    addText(
      'device_context.json',
      const JsonEncoder.withIndent('  ').convert(exportedContext),
    );
    if (runDirectory != null) {
      final manifest = File('${runDirectory!.path}/run_manifest.json');
      if (await manifest.exists()) {
        addText('run_manifest.json', await manifest.readAsString());
      }
      final rows = <Map<String, dynamic>>[],
          sessions = <Map<String, dynamic>>[],
          summaries = <Map<String, dynamic>>[];
      final exportVariants =
          ((exportedContext as Map)['selected_models'] as List?)
              ?.cast<String>() ??
          packet!.variants;
      final exportTasks = _tasks(exportVariants);
      for (final task in exportTasks) {
        final file = File('${runDirectory!.path}/tasks/${task['id']}.json');
        if (!await file.exists()) continue;
        final obj = jsonDecode(await file.readAsString()) as Map;
        final records = (obj['records'] as List)
            .map((r) => Map<String, dynamic>.from(r as Map))
            .toList();
        rows.addAll(records);
        final times =
            records.map((r) => (r['total_ms'] as num).toDouble()).toList()
              ..sort();
        final mid = times.length ~/ 2;
        final median = times.length.isEven
            ? (times[mid - 1] + times[mid]) / 2
            : times[mid];
        summaries.add({
          'profile': obj['profile'],
          'variant': obj['variant'],
          'mode': obj['mode'],
          'measurements': records.length,
          'mean_total_ms': times.reduce((a, b) => a + b) / times.length,
          'median_total_ms': median,
          'p95_total_ms': times[((times.length - 1) * .95).round()],
          'stage2_rate':
              records.where((r) => r['invoked'] == true).length /
              records.length,
          'thermal_state': records.first['thermal_state'],
        });
        for (final s in obj['snapshots'] as List) {
          sessions.add({
            'task_id': obj['task_id'],
            'profile': obj['profile'],
            'variant': obj['variant'],
            'mode': obj['mode'],
            'session_init_ms': obj['session_init_ms'],
            ...Map<String, dynamic>.from(s as Map),
          });
        }
      }
      if (rows.isNotEmpty) {
        addText(
          complete
              ? ((exportedContext['allow_quint8_cross_runtime_drift'] == true)
                    ? 'device_results_QUINT8_PORTABILITY.csv'
                    : exportVariants.length == 1
                    ? 'device_results_FP32_ONLY.csv'
                    : 'device_results.csv')
              : 'device_results_INCOMPLETE.csv',
          csvRows(rows),
        );
      }
      if (summaries.isNotEmpty) {
        addText(
          'timing_summary.csv',
          csvRows(summaries, columns: summaries.first.keys.toList()),
        );
      }
      if (sessions.isNotEmpty) {
        final columns = <String>{};
        for (final row in sessions) {
          columns.addAll(row.keys);
        }
        addText(
          'session_memory_thermal_snapshots.csv',
          csvRows(sessions, columns: columns.toList()),
        );
      }
      addText(
        'completion.json',
        jsonEncode({
          'complete': complete,
          'completed_tasks': summaries.length,
          'required_tasks': exportTasks.length,
          'selected_models': exportVariants,
          'evidence_plan_hash': packet!.evidenceHash,
        }),
      );
    }
    addText(
      'READ_ME.txt',
      'Actual device study logs. QUINT8_PORTABILITY exports measure the frozen QUInt8 model with cross-runtime probability/decision drift retained. They are exploratory device evaluations, not successful Kaggle-equivalence tests; FP32 remains strict and invoked QUInt8 outputs must repeat the captured device reference. A complete FP32-only study has device_results_FP32_ONLY.csv with 8 tasks and 2560 rows. A complete two-model study has device_results.csv with 16 tasks and 5120 rows. FP32-only exports do not validate INT8 portability or speedup. INCOMPLETE files are diagnostics. Memory is current whole-app snapshot: Android PSS in KiB, iOS resident bytes; neither is peak or model-only. UI/network/battery are outside the inference-pipeline timing. Reference mismatches remain failures of Kaggle equivalence, including in completed portability measurements.\n',
    );
    final target = File(
      '${base.path}/crisisq_device_results_${_timestamp()}.zip',
    );
    await target.writeAsBytes(ZipEncoder().encode(archive), flush: true);
    exportPath = target.path;
    notifyListeners();
    return target.path;
  }
}
