# Local model artifacts

Model weights are runtime dependencies, not source code. They are excluded from
Git and from Docker build contexts.

Docker Compose provisions them automatically into the persistent
`openvoice-lab_model-artifacts` volume. Native development can provision the
same verified files with:

```powershell
cd backend
..\.runtime\serving-venv\Scripts\python.exe scripts\download_models.py
..\.runtime\serving-venv\Scripts\python.exe scripts\download_cpu_models.py
```

| Artifact | Role | SHA-256 |
| --- | --- | --- |
| `kokoro-v1.0.onnx` | FP32 model | `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a` |
| `kokoro-v1.0.fp16.onnx` | FP16 model | `f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96` |
| `kokoro-v1.0.int8.onnx` | INT8 quantized model | `ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c` |
| `voices-v1.0.bin` | Shared voice vectors | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` |

The ONNX and voice files come from the upstream
[`kokoro-onnx` `model-files-v1.1` release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1).
The downloader refuses a file whose digest differs from the table above.

## CPU-compatible Audio8 and SpeechT5

`download_cpu_models.py` pins every source revision and every downloaded file
digest. It writes the complete machine-readable inventory to the ignored local
file `cpu-models-provisioned.json` after verification succeeds.

| Runtime | Pinned source | Key verified artifact |
| --- | --- | --- |
| Audio8 INT4 ONNX | `Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4@818569c6b832118ad68d61bbd873abe250fcd68a` | `slow_ar_int4.onnx.data` — `bb217f654039692204386b7e5b74d98e9268863bb664a849aa123a9053d6c824` |
| SpeechT5 | `microsoft/speecht5_tts@30fcde30f19b87502b8435427b5f5068e401d5f6` | `pytorch_model.bin` — `d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190` |
| SpeechT5 vocoder | `microsoft/speecht5_hifigan@bb6f429406e86a9992357a972c0698b22043307d` | `pytorch_model.bin` — `b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b` |
| CMU speaker profile | `Matthijs/cmu-arctic-xvectors@5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5` | selected `cmu-slt.npy` — `21719c0414a470561e6d037466fd239ab59c1f9ed4e1b97db557dad6d0223e73` |

Audio8 uses the upstream no-reference prompt and does not package the optional
registration encoder. The public `unconditioned` voice therefore cannot clone
a submitted speaker. SpeechT5 uses the fixed, checksum-verified `cmu-slt`
speaker profile.

## Licensing

- The [`kokoro-onnx` wrapper and conversion repository](https://github.com/thewh1teagle/kokoro-onnx)
  are MIT licensed.
- The upstream [`Kokoro-82M` model and weights](https://huggingface.co/hexgrad/Kokoro-82M)
  are Apache-2.0 licensed.
- The Audio8 ONNX export and runtime are Apache-2.0 licensed.
- The pinned Microsoft SpeechT5/vocoder repositories and selected CMU x-vector
  dataset are MIT licensed.

Review the upstream licenses for the intended deployment. This repository does
not redistribute the model binaries; it records provenance and downloads them
from the named upstream release at deployment time.
