Supplementary small-packet diagnostic records, not new device classification.

The three ort128_*.tsv files contain source IDs and native-library probabilities
for the original 256-message timing packet on macOS ARM64. The C++ runner is
scripts/native_runtime_probe.cpp. Its CLI takes model path, token-input text
file, profile name, and output TSV. It uses two intra-op threads and one inter-op
thread, distinct from the four-thread device timing protocol. It requires a
locally available matching ONNX Runtime C++ library/headers; no binary is shipped.
The token-input format is source ID followed by 128 IDs and 128 mask integers.

local_probes.json retains Python ONNX Runtime 1.31.1 ARM64 diagnostic outputs,
settings, reference scores and recorded Android scores, without source text.
These fixed records do not reproduce hardware execution on another machine;
the included summary script independently recomputes counts and decision changes.
None of the tested settings is claimed to fix cross-environment agreement.

native_dependency_verification.json records hashes and dependency declarations
from the installed upstream flutter_onnxruntime 1.9.0 Android build.gradle and
iOS podspec. Both specify ONNX Runtime 1.28.0. Upstream package source is available
at https://pub.dev/packages/flutter_onnxruntime/versions/1.9.0 . The full upstream
package and native binary are not relicensed or redistributed here.
