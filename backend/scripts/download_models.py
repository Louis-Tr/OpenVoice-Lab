"""Download and verify the upstream Kokoro v1.0 runtime artifacts."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "model-artifacts"
RELEASE_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)


@dataclass(frozen=True, slots=True)
class Artifact:
    filename: str
    sha256: str


ARTIFACTS = (
    Artifact(
        filename="kokoro-v1.0.onnx",
        sha256="beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
    ),
    Artifact(
        filename="voices-v1.0.bin",
        sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    ),
)


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def provision(artifact: Artifact) -> None:
    destination = ARTIFACT_DIR / artifact.filename
    if destination.exists():
        if digest(destination) == artifact.sha256:
            print(f"verified {artifact.filename}")
            return
        raise RuntimeError(
            f"Refusing to overwrite invalid artifact: {destination}. Remove it and retry."
        )

    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"downloading {artifact.filename}")
    try:
        with (
            urlopen(f"{RELEASE_BASE_URL}/{artifact.filename}", timeout=60) as response,
            temporary.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = digest(temporary)
        if actual != artifact.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {artifact.filename}: expected {artifact.sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {artifact.filename}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        provision(artifact)


if __name__ == "__main__":
    main()
