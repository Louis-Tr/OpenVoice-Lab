# Local model artifacts

Model weights are runtime dependencies, not source code. They are excluded from
Git and from Docker build contexts.

Docker Compose provisions them automatically into the persistent
`openvoice-lab_model-artifacts` volume. Native development can provision the
same verified files with:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\download_models.py
```

| Artifact | Role | SHA-256 |
| --- | --- | --- |
| `kokoro-v1.0.onnx` | FP32 model | `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a` |
| `kokoro-v1.0.int8.onnx` | INT8 quantized model | `ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c` |
| `voices-v1.0.bin` | Shared voice vectors | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` |

The ONNX and voice files come from the upstream
[`kokoro-onnx` `model-files-v1.1` release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1).
The downloader refuses a file whose digest differs from the table above.

## Licensing

- The [`kokoro-onnx` wrapper and conversion repository](https://github.com/thewh1teagle/kokoro-onnx)
  are MIT licensed.
- The upstream [`Kokoro-82M` model and weights](https://huggingface.co/hexgrad/Kokoro-82M)
  are Apache-2.0 licensed.

Review the upstream licenses for the intended deployment. This repository does
not redistribute the model binaries; it records provenance and downloads them
from the named upstream release at deployment time.
