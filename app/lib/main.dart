import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'services/study_controller.dart';
import 'screens/full_evaluation_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CrisisQApp());
}

class CrisisQApp extends StatelessWidget {
  final StudyController? controller;
  final bool initialize;
  const CrisisQApp({super.key, this.controller, this.initialize = true});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'CrisisQ Device Study',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff17557a)),
      useMaterial3: true,
    ),
    home: StudyScreen(controller: controller, initialize: initialize),
  );
}

class StudyScreen extends StatefulWidget {
  final StudyController? controller;
  final bool initialize;
  const StudyScreen({super.key, this.controller, required this.initialize});
  @override
  State<StudyScreen> createState() => _StudyScreenState();
}

class _StudyScreenState extends State<StudyScreen> with WidgetsBindingObserver {
  late final StudyController c;
  final label = TextEditingController();
  bool sharing = false;
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    c = widget.controller ?? StudyController();
    c.addListener(refresh);
    if (widget.initialize) {
      c.initialize().then((_) {
        if (mounted) {
          label.text = c.deviceLabel;
          setState(() {});
        }
      });
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed && c.busy) c.cancel();
  }

  void refresh() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    c.removeListener(refresh);
    if (widget.controller == null) c.dispose();
    label.dispose();
    super.dispose();
  }

  Future<void> export(BuildContext buttonContext) async {
    setState(() => sharing = true);
    try {
      final file = await c.makeExport();
      if (!buttonContext.mounted) return;
      final box = buttonContext.findRenderObject() as RenderBox?;
      await SharePlus.instance.share(
        ShareParams(
          files: [XFile(file)],
          subject: 'CrisisQ device study',
          sharePositionOrigin: box == null
              ? const Rect.fromLTWH(0, 0, 1, 1)
              : box.localToGlobal(Offset.zero) & box.size,
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Export failed: $e')));
      }
    } finally {
      if (mounted) setState(() => sharing = false);
    }
  }

  Widget panel(String title, List<Widget> children) => Card(
    margin: const EdgeInsets.only(bottom: 16),
    child: Padding(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),
          ...children,
        ],
      ),
    ),
  );
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: const Text('CrisisQ · Device study'),
      actions: [
        TextButton(
          onPressed: c.busy || c.packet == null
              ? null
              : () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => FullEvaluationScreen(parent: c),
                  ),
                ),
          child: const Text('Full evaluation'),
        ),
      ],
    ),
    body: SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 780),
          child: ListView(
            padding: const EdgeInsets.all(18),
            children: [
              Text(
                'Measure the frozen pipeline',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 6),
              const Text(
                'Offline research benchmark · 256 fixed messages · two model formats',
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                children: [
                  Chip(
                    label: Text(
                      kReleaseMode ? 'Release build' : 'Setup / debug build',
                    ),
                  ),
                  Chip(
                    label: Text(
                      c.physicalDevice
                          ? 'Physical device'
                          : 'Device check pending / simulator',
                    ),
                  ),
                  const Chip(label: Text('CPU provider')),
                ],
              ),
              if (!kReleaseMode)
                const Padding(
                  padding: EdgeInsets.only(bottom: 14),
                  child: Text(
                    'Use a physical device and flutter run --release for recorded timing. Setup and reference checks work in debug mode.',
                  ),
                ),
              panel('1 · Verify assets', [
                const Text(
                  'Use the exact two ONNX graphs and saved tokenizer from your completed Kaggle output.',
                ),
                const SizedBox(height: 12),
                FilledButton.tonalIcon(
                  onPressed: c.busy || c.packet == null ? null : c.prepare,
                  icon: const Icon(Icons.verified_outlined),
                  label: Text(
                    c.modelsReady ? 'Verify again' : 'Verify model files',
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  c.modelsReady
                      ? 'Model hashes verified.'
                      : 'Model files need verification.',
                ),
                Text(
                  c.tokenizerPassed
                      ? 'Token IDs match all 256 references.'
                      : 'Full-pipeline tokenization requires a reference match.',
                ),
              ]),
              panel('2 · Record the device', [
                Text(
                  c.device['model']?.toString() ??
                      'Loading device information…',
                ),
                Text(c.device['os']?.toString() ?? ''),
                const SizedBox(height: 12),
                TextField(
                  controller: label,
                  enabled: !c.busy,
                  decoration: const InputDecoration(
                    labelText: 'Device label',
                    hintText: 'Samsung A04 · run 1',
                    border: OutlineInputBorder(),
                  ),
                  onChanged: (v) => c.deviceLabel = v,
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<int>(
                  initialValue: c.threads,
                  decoration: const InputDecoration(
                    labelText: 'CPU threads',
                    border: OutlineInputBorder(),
                  ),
                  items: [1, 2, 4]
                      .map((n) => DropdownMenuItem(value: n, child: Text('$n')))
                      .toList(),
                  onChanged: c.busy ? null : (n) => c.configure(threadCount: n),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: c.fullPipeline,
                  onChanged: c.busy
                      ? null
                      : (v) => c.configure(useTokenizer: v),
                  title: const Text('Include on-device tokenization'),
                  subtitle: Text(
                    c.fullPipeline
                        ? 'Routing + tokenization + model'
                        : 'Routing + model (pretokenized)',
                  ),
                ),
                const Text(
                  'Keep settings fixed for the full run. The OS thermal state is recorded for each task.',
                ),
              ]),
              panel('3 · Check frozen predictions', [
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: c.fp32Only,
                  onChanged: c.busy ? null : (v) => c.configure(onlyFp32: v),
                  title: const Text('FP32-only study'),
                  subtitle: const Text(
                    'Enabled: 8 tasks, 2,560 rows. Disable for a two-model comparison after choosing the appropriate QUInt8 validation mode.',
                  ),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  value: c.allowQuint8Drift,
                  onChanged: c.busy
                      ? null
                      : (v) => c.configure(permitQuint8Drift: v),
                  title: const Text('QUInt8 portability study'),
                  subtitle: const Text(
                    'Turn FP32-only off to compare both models. Records Kaggle drift rather than claiming equivalence. Requires strict FP32 checks and repeatable QUInt8 device outputs.',
                  ),
                ),
                if (c.quint8Portability)
                  const Text(
                    'Exploratory device study: original model and threshold stay frozen. QUInt8 probability and decision changes are reported. This does not establish preserved accuracy or successful Kaggle equivalence.',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                const Text(
                  'Reference checks run both models. Timed QUInt8 outputs are checked against the captured device outputs in portability mode.',
                ),
                const SizedBox(height: 12),
                FilledButton.tonalIcon(
                  onPressed: c.busy || !c.modelsReady
                      ? null
                      : c.checkReferences,
                  icon: const Icon(Icons.fact_check_outlined),
                  label: const Text('Check references'),
                ),
                const SizedBox(height: 8),
                Text(
                  c.selectedModelsReady
                      ? (c.quint8Portability
                            ? 'Portability measurements ready. QUInt8 Kaggle equivalence is reported separately.'
                            : 'Selected model reference checks passed.')
                      : 'Selected model checks pending or failed.',
                ),
                const SizedBox(height: 8),
                Text(c.quint8DiagnosticSummary),
              ]),
              panel('4 · Run and export', [
                Text(
                  'Four profiles × ${c.fp32Only ? 'FP32 only' : 'FP32/INT8'} × neural/hybrid. Each task uses 10 warmups and five repeats of the same 64 messages.',
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    FilledButton.icon(
                      onPressed: c.canBenchmark ? () => c.runStudy() : null,
                      icon: const Icon(Icons.play_arrow),
                      label: Text(
                        c.fp32Only
                            ? 'Start FP32 study'
                            : c.quint8Portability
                            ? 'Start QUInt8 comparison'
                            : 'Start full study',
                      ),
                    ),
                    OutlinedButton(
                      onPressed: c.canBenchmark
                          ? () => c.runStudy(resume: true)
                          : null,
                      child: const Text('Resume saved study'),
                    ),
                    if (c.busy)
                      OutlinedButton(
                        onPressed: c.cancel,
                        child: const Text('Pause'),
                      ),
                  ],
                ),
                const SizedBox(height: 12),
                if (c.busy)
                  LinearProgressIndicator(
                    value: c.progress > 0 ? c.progress : null,
                  ),
                const SizedBox(height: 8),
                Text(
                  c.status,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                if (c.completedTasks > 0)
                  Text('${c.completedTasks}/${c.studyTaskCount} tasks saved'),
                if (c.error.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 10),
                    child: SelectableText(
                      c.error,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                      ),
                    ),
                  ),
                const SizedBox(height: 12),
                Builder(
                  builder: (ctx) => FilledButton.tonalIcon(
                    onPressed:
                        c.busy ||
                            sharing ||
                            (c.parityReport.isEmpty && c.runDirectory == null)
                        ? null
                        : () => export(ctx),
                    icon: const Icon(Icons.ios_share),
                    label: Text(
                      sharing
                          ? 'Preparing export…'
                          : c.complete
                          ? 'Export complete results'
                          : 'Export diagnostics / saved tasks',
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  'The ZIP contains CSV logs and context JSON. Model weights and source texts are excluded.',
                ),
              ]),
              const Text(
                'Memory values are current whole-app snapshots. Android PSS and iOS RSS differ; neither is a peak-memory measurement.',
              ),
              const SizedBox(height: 12),
              Text(
                'Runtime: ${StudyController.runtimeLabel}\nPlatform: ${Platform.operatingSystem}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    ),
  );
}
