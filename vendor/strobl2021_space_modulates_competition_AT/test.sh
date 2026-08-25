#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$ROOT/test_released_jar.sh"
"$ROOT/build.sh"

mkdir -p "$ROOT/.build/test-classes"
javac -encoding UTF-8 -source 8 -target 8 \
    -cp "$ROOT/.build/classes" \
    -d "$ROOT/.build/test-classes" \
    "$ROOT/tests/ControlledModelTest.java"
java -ea -cp "$ROOT/.build/classes:$ROOT/.build/test-classes" \
    strobl.control.ControlledModelTest

java -jar "$ROOT/controlled-model.jar" \
    --mode batch \
    --width 10 \
    --height 8 \
    --family two_resistant_nests \
    --sensitive 36 \
    --resistant 6 \
    --simulation-seed 19 \
    --ic-seed 23 \
    --steps 6 \
    --dose 0.75 \
    --out "$ROOT/.build/controlled_batch_smoke.csv"
cmp "$ROOT/fixtures/controlled_batch_smoke.csv" \
    "$ROOT/.build/controlled_batch_smoke.csv"

protocol_output="$(
    printf 'RESET resistant_core 12 3 5 7\nSTEP 1\nGRID\nQUIT\n' \
        | java -jar "$ROOT/controlled-model.jar" \
            --mode serve --width 5 --height 4
)"
printf '%s\n' "$protocol_output" | rg '^READY[[:space:]]+strobl-controlled-v1[[:space:]]+aa3b3c2ad2e4acf9fd7cc6ac318f1bf79f9361e2$' >/dev/null
printf '%s\n' "$protocol_output" | rg '^STATE[[:space:]]+step=1' >/dev/null
printf '%s\n' "$protocol_output" | rg '^GRID[[:space:]]+width=5[[:space:]]+height=4' >/dev/null
printf '%s\n' "$protocol_output" | rg '^BYE$' >/dev/null

echo "Regression and protocol smokes: PASS"
