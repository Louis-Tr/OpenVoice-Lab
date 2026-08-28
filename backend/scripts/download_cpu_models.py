"""Provision checksum-pinned CPU model artifacts for Audio8 and SpeechT5."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "model-artifacts"

AUDIO8_REPOSITORY = "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4"
AUDIO8_REVISION = "818569c6b832118ad68d61bbd873abe250fcd68a"
SPEECHT5_REPOSITORY = "microsoft/speecht5_tts"
SPEECHT5_REVISION = "30fcde30f19b87502b8435427b5f5068e401d5f6"
VOCODER_REPOSITORY = "microsoft/speecht5_hifigan"
VOCODER_REVISION = "bb6f429406e86a9992357a972c0698b22043307d"
SPEAKER_REPOSITORY = "datasets/Matthijs/cmu-arctic-xvectors"
SPEAKER_REVISION = "5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5"


@dataclass(frozen=True, slots=True)
class RemoteArtifact:
    repository: str
    revision: str
    remote_path: str
    local_path: str
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{self.remote_path}?download=true"
        )


ARTIFACTS = (
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "codec_decoder_fp16.onnx",
        "audio8-tts-preview-0.6b-int4/codec_decoder_fp16.onnx",
        "6e379be31db6c1b0c111e0e3d2aeb10717ee96b197462b926de411e75a1fd019",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "codec_decoder_fp16.onnx.data",
        "audio8-tts-preview-0.6b-int4/codec_decoder_fp16.onnx.data",
        "18838f686aa7c1528fb69ec11e1ab404fdc4dc823d13219abfd4b327988527c0",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "fast_ar_int4.onnx",
        "audio8-tts-preview-0.6b-int4/fast_ar_int4.onnx",
        "808c5a0c95c28d90337d925a9a8f6075f7ff8eb7b3080d2b34c4133479a6dc94",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "fast_ar_int4.onnx.data",
        "audio8-tts-preview-0.6b-int4/fast_ar_int4.onnx.data",
        "183be0c9f26b27c605b92a0875beb93f8f98b771f27f65cab133c73610868325",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "slow_ar_int4.onnx",
        "audio8-tts-preview-0.6b-int4/slow_ar_int4.onnx",
        "0cf7701d6da81f888b49ba6e752445d9786a9915ba30dcf084f7743bdda96834",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "slow_ar_int4.onnx.data",
        "audio8-tts-preview-0.6b-int4/slow_ar_int4.onnx.data",
        "bb217f654039692204386b7e5b74d98e9268863bb664a849aa123a9053d6c824",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "runtime_manifest.json",
        "audio8-tts-preview-0.6b-int4/runtime_manifest.json",
        "6473ae7d0106a2e369e442c72a71d2d46d8fbd3fe18c80d80b1b46e4aa241930",
        "Apache-2.0",
    ),
    RemoteArtifact(
        AUDIO8_REPOSITORY,
        AUDIO8_REVISION,
        "tokenizer/tokenizer.json",
        "audio8-tts-preview-0.6b-int4/tokenizer/tokenizer.json",
        "f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b",
        "Apache-2.0",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "added_tokens.json",
        "speecht5-tts/added_tokens.json",
        "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad",
        "MIT",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "config.json",
        "speecht5-tts/config.json",
        "2caf62dde93699a90cfc35ff2a8de27b02b479a0c98881cbc55f9682cc43e258",
        "MIT",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "preprocessor_config.json",
        "speecht5-tts/preprocessor_config.json",
        "9461d890ff65badba5ad726f9059436bc69963883dc803fa6ed3cdb8f8af3687",
        "MIT",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "special_tokens_map.json",
        "speecht5-tts/special_tokens_map.json",
        "2a098b61fe8ec4cfd7674832ca00b4268c07569743a4ad15c8164e8f60ebf981",
        "MIT",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "tokenizer_config.json",
        "speecht5-tts/tokenizer_config.json",
        "d589430c619db2d95ff0fa757a187b55ef5ea44eff7fb08a6fbf0e78e32a6247",
        "MIT",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "spm_char.model",
        "speecht5-tts/spm_char.model",
        "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560",
        "MIT",
    ),
    RemoteArtifact(
        SPEECHT5_REPOSITORY,
        SPEECHT5_REVISION,
        "pytorch_model.bin",
        "speecht5-tts/pytorch_model.bin",
        "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190",
        "MIT",
    ),
    RemoteArtifact(
        VOCODER_REPOSITORY,
        VOCODER_REVISION,
        "config.json",
        "speecht5-hifigan/config.json",
        "ac281bbb65c617a3fe7c5c082c8106f5a35b14d236b6a12f99cbbb12e576ab96",
        "MIT",
    ),
    RemoteArtifact(
        VOCODER_REPOSITORY,
        VOCODER_REVISION,
        "pytorch_model.bin",
        "speecht5-hifigan/pytorch_model.bin",
        "b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b",
        "MIT",
    ),
)

SPEAKER_ARCHIVE_SHA256 = "28ea1b685a49fedce92d1af7e68b22bf511a23432bc7a13d621a4deeee9fe9a1"
SPEAKER_MEMBER = "spkrec-xvect/cmu_us_slt_arctic-wav-arctic_a0001.npy"
SPEAKER_SHA256 = "21719c0414a470561e6d037466fd239ab59c1f9ed4e1b97db557dad6d0223e73"


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "OpenVoice-Lab-model-provisioner/1.0"})
    with urlopen(request, timeout=120) as response, destination.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def provision(artifact: RemoteArtifact) -> None:
    destination = ARTIFACT_ROOT / artifact.local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) == artifact.sha256:
            print(f"verified {artifact.local_path}")
            return
        raise RuntimeError(
            f"Refusing to overwrite invalid artifact: {destination}. Remove it and retry."
        )
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"downloading {artifact.repository}/{artifact.remote_path}")
    try:
        _download(artifact.url, temporary)
        actual = digest(temporary)
        if actual != artifact.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {artifact.remote_path}: "
                f"expected {artifact.sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {artifact.local_path}")


def provision_speaker_embedding() -> None:
    destination = ARTIFACT_ROOT / "speecht5-speakers" / "cmu-slt.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) == SPEAKER_SHA256:
            print("verified speecht5-speakers/cmu-slt.npy")
            return
        raise RuntimeError(
            f"Refusing to overwrite invalid speaker profile: {destination}. Remove it and retry."
        )
    url = (
        f"https://huggingface.co/{SPEAKER_REPOSITORY}/resolve/"
        f"{SPEAKER_REVISION}/spkrec-xvect.zip?download=true"
    )
    with tempfile.TemporaryDirectory(prefix="openvoice-speaker-") as directory:
        archive = Path(directory) / "speaker.zip"
        _download(url, archive)
        actual_archive = digest(archive)
        if actual_archive != SPEAKER_ARCHIVE_SHA256:
            raise RuntimeError(
                "Checksum mismatch for SpeechT5 speaker archive: "
                f"expected {SPEAKER_ARCHIVE_SHA256}, got {actual_archive}"
            )
        with zipfile.ZipFile(archive) as source:
            member = source.getinfo(SPEAKER_MEMBER)
            if PurePosixPath(member.filename) != PurePosixPath(SPEAKER_MEMBER):
                raise RuntimeError("Unexpected speaker archive member path")
            temporary = destination.with_suffix(".npy.download")
            try:
                with source.open(member) as input_file, temporary.open("xb") as output:
                    while chunk := input_file.read(1024 * 1024):
                        output.write(chunk)
                actual = digest(temporary)
                if actual != SPEAKER_SHA256:
                    raise RuntimeError(
                        "Checksum mismatch for selected speaker profile: "
                        f"expected {SPEAKER_SHA256}, got {actual}"
                    )
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    print("verified speecht5-speakers/cmu-slt.npy")


def write_provenance_marker() -> None:
    marker = ARTIFACT_ROOT / "cpu-models-provisioned.json"
    payload = {
        "schemaVersion": 1,
        "artifacts": [asdict(artifact) for artifact in ARTIFACTS],
        "speakerProfile": {
            "repository": SPEAKER_REPOSITORY,
            "revision": SPEAKER_REVISION,
            "archivePath": "spkrec-xvect.zip",
            "archiveSha256": SPEAKER_ARCHIVE_SHA256,
            "member": SPEAKER_MEMBER,
            "localPath": "speecht5-speakers/cmu-slt.npy",
            "sha256": SPEAKER_SHA256,
            "license": "MIT",
        },
    }
    temporary = marker.with_suffix(".json.download")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    print("wrote cpu-models-provisioned.json")


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        provision(artifact)
    provision_speaker_embedding()
    write_provenance_marker()


if __name__ == "__main__":
    main()
