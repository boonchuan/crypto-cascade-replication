"""make_manifest.py — SHA-256 manifest of the data vintage and outputs.

Usage: python -m src.make_manifest
Writes MANIFEST.sha256 at the repo root; commit it and attach it to the
release so the archived vintage is verifiable.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = []
    for folder in ["data", "output"]:
        for p in sorted((ROOT / folder).glob("*")):
            if p.is_file():
                lines.append(f"{sha256(p)}  {folder}/{p.name}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
    print(f"MANIFEST.sha256 written ({len(lines)} files)")


if __name__ == "__main__":
    main()
