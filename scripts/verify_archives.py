import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Verify downloaded Apex reviewer archives")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    index_path = Path(__file__).resolve().parents[1] / "docs" / "ARCHIVE_INDEX.json"
    index = json.loads(index_path.read_text())
    for item in index:
        path = args.directory / item["file"]
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {path.name}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"ZIP CRC failure: {path.name}")
            sum_paths = [name for name in archive.namelist() if name.endswith("/SHA256SUMS.txt")]
            if len(sum_paths) != 1:
                raise ValueError(f"Expected one internal checksum index: {path.name}")
            members = set()
            for line in archive.read(sum_paths[0]).decode().splitlines():
                expected, name = line.split("  ", 1)
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Member SHA-256 mismatch: {name}")
                members.add(name)
            if members | set(sum_paths) != set(archive.namelist()):
                raise ValueError(f"Unindexed archive members: {path.name}")
        print(f"Verified {path.name}: archive SHA-256, ZIP CRC, and {len(members)} member hashes")


if __name__ == "__main__":
    main()
