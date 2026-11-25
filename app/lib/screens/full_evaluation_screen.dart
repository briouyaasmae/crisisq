import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import '../services/study_controller.dart';
import '../services/full_evaluation_controller.dart';

class FullEvaluationScreen extends StatefulWidget {
  final StudyController parent;
  const FullEvaluationScreen({super.key, required this.parent});
  @override
  State<FullEvaluationScreen> createState() => _FullEvaluationScreenState();
}

class _FullEvaluationScreenState extends State<FullEvaluationScreen>
    with WidgetsBindingObserver {
  late final FullEvaluationController c;
  bool sharing = false;
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    c = FullEvaluationController(widget.parent);
    c.addListener(_refresh);
    c.initialize();
  }

  void _refresh() {
    if (mounted) setState(() {});
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed && c.busy) c.cancel();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    c.removeListener(_refresh);
    c.dispose();
    super.dispose();
  }

  Future<void> _export(BuildContext buttonContext) async {
    setState(() => sharing = true);
    try {
      final path = await c.makeExport();
      if (!buttonContext.mounted) return;
      final box = buttonContext.findRenderObject() as RenderBox?;
      await SharePlus.instance.share(
        ShareParams(
          files: [XFile(path)],
          subject: 'CrisisQ full labeled evaluation',
          sharePositionOrigin: box == null
              ? const Rect.fromLTWH(0, 0, 1, 1)
              : box.localToGlobal(Offset.zero) & box.size,
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(e.toString())));
      }
    } finally {
      if (mounted) setState(() => sharing = false);
    }
  }

  @override
  Widget build(BuildContext context) => PopScope(
    canPop: !c.busy,
    child: Scaffold(
      appBar: AppBar(title: const Text('Full labeled evaluation')),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 780),
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
                Text(
                  'Classify all 1,700 frozen labeled messages',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 12),
                const Text(
                  '1,000 binary-test + 300 hard-negative + 400 probe examples. The same frozen models, rules, labels and threshold are used. Unlabeled support data is excluded.',
                ),
                const SizedBox(height: 12),
                const Text(
                  'Both FP32 and QUInt8 are evaluated. Each model runs once per message; neural-only and hybrid decisions are derived from that probability and the frozen gate. This produces 3,400 model outputs and 6,800 decisions across four configurations.',
                ),
                const SizedBox(height: 12),
                const Text(
                  'Classification only: saved tokenizer inputs are used. This screen does not measure pipeline latency or create a new independent test set. QUInt8 Kaggle differences remain reported.',
                ),
                const Divider(height: 32),
                Text(
                  'Device: ${c.parent.deviceLabel} · ${c.parent.threads} CPU threads',
                ),
                Text(
                  c.parent.variantPassed['fp32'] == true
                      ? '256-case FP32 preflight passed.'
                      : 'First verify files and pass reference checks on the setup screen.',
                ),
                FilledButton.tonal(
                  onPressed: c.busy || c.parent.busy || !c.parent.modelsReady
                      ? null
                      : c.parent.checkReferences,
                  child: const Text('Run setup reference checks'),
                ),
                const Text(
                  'Let the device cool after reference checks. Thermal guards remain enabled. Use two threads if four repeatedly causes iPhone thermal pauses; choose threads on the setup screen before starting.',
                ),
                const SizedBox(height: 16),
                Wrap(
                  spacing: 12,
                  runSpacing: 12,
                  children: [
                    FilledButton(
                      onPressed: c.canStart ? () => c.run() : null,
                      child: const Text('Start full evaluation'),
                    ),
                    OutlinedButton(
                      onPressed: c.canStart ? () => c.run(resume: true) : null,
                      child: const Text('Resume full evaluation'),
                    ),
                    if (c.busy)
                      OutlinedButton(
                        onPressed: c.cancel,
                        child: const Text('Pause'),
                      ),
                  ],
                ),
                const SizedBox(height: 16),
                if (c.busy)
                  LinearProgressIndicator(value: c.savedRows / c.requiredRows),
                const SizedBox(height: 12),
                Text(
                  c.status,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                if (c.error.isNotEmpty)
                  SelectableText(
                    c.error,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                Text('${c.savedRows}/${c.requiredRows} model outputs saved'),
                const SizedBox(height: 16),
                Builder(
                  builder: (ctx) => FilledButton.tonalIcon(
                    onPressed: c.busy || sharing || c.runDirectory == null
                        ? null
                        : () => _export(ctx),
                    icon: const Icon(Icons.ios_share),
                    label: Text(
                      sharing
                          ? 'Preparing ZIP…'
                          : c.complete
                          ? 'Export full evaluation'
                          : 'Export saved partial results',
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Keep this screen foreground while running. It pauses after backgrounding; incomplete chunks restart on resume. A five-second rest is inserted between 25-message checkpoints. Cool longer if the OS reports serious or severe heat. Pause before closing this screen.',
                ),
                const SizedBox(height: 12),
                const Text(
                  'The export contains predictions, source labels, group IDs, reference differences and context. It excludes source text and model weights. Source labels are nonclinical.',
                ),
              ],
            ),
          ),
        ),
      ),
    ),
  );
}
