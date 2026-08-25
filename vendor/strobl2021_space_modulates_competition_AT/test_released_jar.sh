#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_JAR_SHA256="42cb0b7cba654cfe2297c47d13285ffdc143a0554ed75b75ad40cc1a48ad3983"
OUTPUT_DIR="$ROOT/.build/released-jar-smoke"
OUTPUT_FILE="$OUTPUT_DIR/AT50_cellCounts_cost_0.0_rFrac_0.1_initSize_0.02_dt_1.0_RepId_7.csv"

actual_sha256="$(shasum -a 256 "$ROOT/upstream/onLatticeModel.jar" | awk '{print $1}')"
test "$actual_sha256" = "$EXPECTED_JAR_SHA256"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
java -Djava.awt.headless=true -jar "$ROOT/upstream/onLatticeModel.jar" \
    -initialSize 0.02 \
    -rFrac 0.1 \
    -turnover 0 \
    -cost 0 \
    -tEnd 2 \
    -seed 7 \
    -nReplicates 1 \
    -compareToMTD false \
    -profilingMode false \
    -terminateAtProgression false \
    -outDir "$OUTPUT_DIR/"

cmp "$ROOT/fixtures/released_jar_smoke.csv" "$OUTPUT_FILE"
echo "Released JAR golden smoke: PASS"
