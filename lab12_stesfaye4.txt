#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="stesfaye4"
INPUT="aminegg.txt"
OUT="${ACCOUNT}_aminegg.txt"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing $INPUT in $(pwd)" >&2
  exit 1
fi

echo "==== Part 1: sed experiments ===="
echo "--- sed 's/ he/ HE/' ---"
sed 's/ he/ HE/' < "$INPUT"
echo "--- sed 's/ he/ HE/g' ---"
sed 's/ he/ HE/g' < "$INPUT"
echo "--- sed 's/^/AAA/' ---"
sed 's/^/AAA/' < "$INPUT"
echo "--- sed 's/$/ZZZ/' ---"
sed 's/$/ZZZ/' < "$INPUT"

echo "==== Add START/EOL, then filter STARTEOL ===="
sed 's/^/START/' "$INPUT" | sed 's/$/EOL/' | grep -v STARTEOL | tee "$OUT" >/dev/null

echo "==== cmp original vs ${OUT} ===="
cmp "$INPUT" "$OUT" || true
echo "cmp: prints first differing byte/line; no output means identical. Options: -s silent, -l list diffs, -i N skip bytes."

echo "==== diff original vs ${OUT} ===="
diff -u "$INPUT" "$OUT" || true

echo "==== Remove anchored START/EOL (display only) ===="
sed 's/^/START/' "$INPUT" | sed 's/$/EOL/' | grep -v STARTEOL | sed 's/^START//' | sed 's/EOL$//' | cat

echo "==== Overwrite ${OUT} with START/EOL-removed content ===="
tmp="$(mktemp)"
sed 's/^/START/' "$INPUT" | sed 's/$/EOL/' | grep -v STARTEOL | sed 's/^START//' | sed 's/EOL$//' > "$tmp"
mv "$tmp" "$OUT"

echo "==== diff original vs ${OUT} after removal ===="
diff -u "$INPUT" "$OUT" || true
echo "--- Explanation of first 3 diff header lines ---"
echo "--- FILE1 timestamp"
echo "+++ FILE2 timestamp"
echo "@@ -L1,LN +R1,RN @@ (hunk ranges)"

echo "==== Part 2: date/time, counts, path, listing ===="
date
find . -maxdepth 1 -type f -not -name '.*' | wc -l
pwd
ls -la