#!/usr/bin/env bash
# Build tagged (PDF/UA-2) PDFs with LuaLaTeX, two passes each, then run the
# structural accessibility check.
#
#   tools/build_pdf.sh source/homeworks/HW1.tex source/homeworks/HW1_sol.tex
#
# Each PDF is written next to its .tex file. Adapted from the DSAI 601
# source/accessible/build.sh.

set -uo pipefail
[ $# -gt 0 ] || { echo "usage: $0 file.tex [file.tex ...]" >&2; exit 2; }
command -v lualatex >/dev/null || { echo "ERROR: lualatex not found on PATH." >&2; exit 1; }

here="$(cd "$(dirname "$0")" && pwd)"
status=0
pdfs=()

for tex in "$@"; do
  [ -f "$tex" ] || { echo "SKIP $tex (not found)"; continue; }
  dir="$(dirname "$tex")"; base="$(basename "$tex" .tex)"
  echo "=== $tex ==="
  for pass in 1 2; do
    (cd "$dir" && lualatex -interaction=nonstopmode -file-line-error "$base.tex" > "$base.pass$pass.log" 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "  pass $pass FAILED (exit $rc). First errors:"
      grep -E '^[^ ]+:[0-9]+:|^! ' "$dir/$base.pass$pass.log" | head -8 | sed 's/^/    /'
      status=1
      break
    fi
    echo "  pass $pass ok"
  done
  if [ -f "$dir/$base.pdf" ]; then
    warns=$(grep -ciE 'tagpdf (warning|error)' "$dir/$base.pass2.log" 2>/dev/null || echo 0)
    echo "  tagpdf warnings: $warns"
    pdfs+=("$dir/$base.pdf")
  fi
  rm -f "$dir/$base.pass1.log"
done

if [ ${#pdfs[@]} -gt 0 ]; then
  echo
  echo "=== accessibility check ==="
  uv run --with pypdf python "$here/check_accessibility.py" "${pdfs[@]}" || status=1
fi
exit $status
