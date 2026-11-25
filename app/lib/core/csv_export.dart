const deviceCsvColumns = <String>[
  'device_id',
  'os',
  'runtime',
  'provider',
  'threads',
  'thermal_state',
  'evidence_plan_hash',
  'profile',
  'source_id',
  'variant',
  'mode',
  'tokenizer_mode',
  'timing_scope',
  'repeat',
  'sequence',
  'total_ms',
  'gate_ms',
  'tokenization_ms',
  'onnx_ms',
  'invoked',
  'probability',
  'decision',
  'parity_verified',
  'run_id',
  'task_id',
  'task_started_utc',
  'build_mode',
  'validation_scope',
  'neural_reference_probability',
  'kaggle_probability_match',
  'device_reference_probability',
  'device_repeatability_verified',
];
String csvField(Object? value) {
  if (value == null) return '';
  final text = value.toString();
  return text.contains(RegExp('[,"\r\n]'))
      ? '"${text.replaceAll('"', '""')}"'
      : text;
}

String csvRows(
  List<Map<String, dynamic>> rows, {
  List<String> columns = deviceCsvColumns,
  bool header = true,
}) {
  final b = StringBuffer();
  if (header) b.writeln(columns.map(csvField).join(','));
  for (final row in rows) {
    b.writeln(columns.map((key) => csvField(row[key])).join(','));
  }
  return b.toString();
}
