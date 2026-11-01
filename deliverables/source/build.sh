#!/usr/bin/env bash
# Render the HTML sources in this folder to PDF with headless Chrome.
# Usage: deliverables/source/build.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$(dirname "$here")"
for name in study_guide technical_handover; do
  google-chrome --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$out/$name.pdf" "file://$here/$name.html" 2>/dev/null
  echo "wrote $out/$name.pdf"
done
