#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/.build"
CLASSES_DIR="$BUILD_DIR/classes"
OUTPUT_JAR="$ROOT/controlled-model.jar"

rm -rf "$CLASSES_DIR"
mkdir -p "$CLASSES_DIR"

sources=("$ROOT"/src/strobl/control/*.java)
javac -encoding UTF-8 -source 8 -target 8 -d "$CLASSES_DIR" "${sources[@]}"

# Python's zip writer gives every entry a fixed timestamp and sorted order,
# making the JAR byte-for-byte reproducible across repeated builds.
python3 - "$CLASSES_DIR" "$OUTPUT_JAR" <<'PY'
from pathlib import Path
import stat
import sys
import zipfile

classes = Path(sys.argv[1])
output = Path(sys.argv[2])
timestamp = (2000, 1, 1, 0, 0, 0)
manifest = (
    b"Manifest-Version: 1.0\r\n"
    b"Main-Class: strobl.control.ControlledCli\r\n"
    b"\r\n"
)

def write_entry(archive, name, data):
    info = zipfile.ZipInfo(name, timestamp)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, data)

with zipfile.ZipFile(output, "w") as archive:
    write_entry(archive, "META-INF/MANIFEST.MF", manifest)
    for path in sorted(classes.rglob("*.class")):
        write_entry(archive, path.relative_to(classes).as_posix(), path.read_bytes())
PY

echo "Built $OUTPUT_JAR"
shasum -a 256 "$OUTPUT_JAR"
