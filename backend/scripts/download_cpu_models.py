"""Provision checksum-pinned CPU model artifacts for SpeechT5."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "model-artifacts"

SPEECHT5_REPOSITORY = "microsoft/speecht5_tts"
SPEECHT5_REVISION = "30fcde30f19b87502b8435427b5f5068e401d5f6"
VOCODER_REPOSITORY = "microsoft/speecht5_hifigan"
VOCODER_REVISION = "bb6f429406e86a9992357a972c0698b22043307d"
SPEAKER_REPOSITORY = "datasets/Matthijs/cmu-arctic-xvectors"
SPEAKER_REVISION = "5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5"
ASR_REPOSITORY = "openai/whisper-small.en"
ASR_REVISION = "e8727524f962ee844a7319d92be39ac1bd25655a"


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
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "added_tokens.json",
        "whisper-small-en/added_tokens.json",
        "560be47bea388757f8d4cc185c5d82067426cbb6361e38016dd90ddc01ab203a",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "config.json",
        "whisper-small-en/config.json",
        "3441094addfb2a31ef62ea117e95bd6e437bbad8e93d3697fe420a4a6d3603a5",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "generation_config.json",
        "whisper-small-en/generation_config.json",
        "aad4e18c0fb1d063c5b38b4080cec3c4c8cdfff95e1de4797013470654a5a72a",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "merges.txt",
        "whisper-small-en/merges.txt",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "model.safetensors",
        "whisper-small-en/model.safetensors",
        "6014ac49b506df900f66f4aca6b0801eed7245594ace97bcaf73e0ae5b863066",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "normalizer.json",
        "whisper-small-en/normalizer.json",
        "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "preprocessor_config.json",
        "whisper-small-en/preprocessor_config.json",
        "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "special_tokens_map.json",
        "whisper-small-en/special_tokens_map.json",
        "014f8f802366ed818919550be0ad9e35907327cb9e142e8aaa102420f460bda8",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "tokenizer.json",
        "whisper-small-en/tokenizer.json",
        "5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "tokenizer_config.json",
        "whisper-small-en/tokenizer_config.json",
        "14f84bdf4b9ecdbd4738ddc81c17a1baedfc02bb93c6e049c951e15a1b40b70d",
        "Apache-2.0",
    ),
    RemoteArtifact(
        ASR_REPOSITORY,
        ASR_REVISION,
        "vocab.json",
        "whisper-small-en/vocab.json",
        "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7",
        "Apache-2.0",
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
