# Flutter device use

Completed comparative evidence is Android only. iOS integration is supplied, but
thermal pauses are not a completed comparative iOS study.

```sh
python scripts/materialize_public_assets.py
python app/tools/prepare_assets.py --source /path/to/whole_frozen_model_output
# Alternatively: --zip /path/to/crisisq_flutter_model_assets.zip
cd app
flutter pub get
flutter analyze
flutter test
flutter devices
flutter run --release -d YOUR_DEVICE_ID
```

The import helper checks `manifests/primary_model_files.json`. Required files are
`fp32.onnx`, `q_quint8.onnx`, `tokenizer.json`, `tokenizer_config.json`, `config.json`.
The model-export notebook in `app/tools/` prepares them from the original whole
model-containing output without retraining. Weights are not supplied publicly.

The previously checked toolchain is Flutter 3.44.6 / Dart 3.12.2, Android API 24+ or
iOS 16+ with relevant platform tools. iOS requires macOS/Xcode/signing. Study records
require release builds on physical devices; debug/simulator runs cannot record one.

On setup, verify assets/references, retain the fixed thread settings and let the
device cool. QUInt8 portability mode preserves failed Kaggle-reference checks and
requires repeatable device outputs; it does not declare equivalence. Full evaluation
scores the 1,700-message packet using both models. Thermal/foreground guards and
checkpoints remain active. Do not change rules or thresholds during comparisons.

Running a phone produces new evidence; repository verification instead checks the
recorded exports. Whole-app PSS is not peak/model-only memory. Host fallback timing
does not measure phone mitigation costs.
