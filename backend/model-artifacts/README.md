# Local model artifacts

Model weights stay on the host and are never committed. Provision the verified
Kokoro v1.0 assets with:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\download_models.py
```

| Artifact | SHA-256 |
| --- | --- |
| `kokoro-v1.0.onnx` | `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a` |
| `voices-v1.0.bin` | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` |

Both files come from the upstream
[`kokoro-onnx` `model-files-v1.1` release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1).
The wrapper is MIT-licensed and the Kokoro model is Apache-2.0 licensed.
