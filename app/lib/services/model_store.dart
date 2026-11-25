import 'dart:io';
import 'package:flutter/services.dart';
import 'package:crypto/crypto.dart';
import 'package:path_provider/path_provider.dart';
import '../models/study_packet.dart';

class MissingModels implements Exception {
  final List<String> names;
  MissingModels(this.names);
  @override
  String toString() =>
      'Add the frozen model assets first: ${names.join(', ')}. See README section 1; then rebuild the app.';
}

class ModelStore {
  final StudyPacket packet;
  Directory? directory;
  ModelStore(this.packet);
  Future<void> prepare(void Function(String) progress) async {
    final assets = await AssetManifest.loadFromAssetBundle(rootBundle);
    final missing = packet.expectedFiles.keys
        .where((n) => !assets.listAssets().contains('assets/models/$n'))
        .toList();
    if (missing.isNotEmpty) throw MissingModels(missing);
    final support = await getApplicationSupportDirectory();
    final dir = Directory('${support.path}/frozen_models');
    await dir.create(recursive: true);
    for (final entry in packet.expectedFiles.entries) {
      progress('Verifying ${entry.key}');
      final file = File('${dir.path}/${entry.key}');
      var valid = false;
      if (await file.exists()) {
        valid =
            (await sha256.bind(file.openRead()).first).toString() ==
            entry.value;
      }
      if (!valid) {
        final data = await rootBundle.load('assets/models/${entry.key}');
        final bytes = data.buffer.asUint8List(
          data.offsetInBytes,
          data.lengthInBytes,
        );
        if (sha256.convert(bytes).toString() != entry.value) {
          throw StateError(
            'Wrong frozen file: ${entry.key}. No substitution is allowed.',
          );
        }
        final tmp = File('${file.path}.partial');
        await tmp.writeAsBytes(bytes, flush: true);
        if (await file.exists()) await file.delete();
        await tmp.rename(file.path);
      }
    }
    directory = dir;
  }

  File file(String name) {
    if (directory == null) throw StateError('Model store is not ready');
    return File('${directory!.path}/$name');
  }
}
