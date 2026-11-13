#!/usr/bin/env bash
# Turn the SVG frames from scripts/demo_capture.py into PNG stills and the
# looping GIF used in the README. Needs rsvg-convert + ImageMagick.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
demo="$here/../docs/demo"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

python3 - "$demo" "$work" <<'PY'
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for svg in sorted(src.glob("f*.svg")):
    text = svg.read_text(encoding="utf-8")
    # Rich exports non-breaking spaces and leaves whitespace collapsing to the
    # renderer; make the spacing explicit so the PNG matches the terminal.
    text = text.replace(
        '<svg class="rich-terminal"',
        '<svg xml:space="preserve" class="rich-terminal"',
    ).replace("&#160;", " ")
    (dst / svg.name).write_text(text, encoding="utf-8")
PY

for svg in "$work"/f*.svg; do
    rsvg-convert -w 1200 "$svg" -o "${svg%.svg}.png"
done

frames=("$work"/f*.png)
count=${#frames[@]}
last="${frames[$((count - 1))]}"

cp "${frames[1]}" "$demo/startup.png"
cp "${frames[2]}" "$demo/indexed.png"
cp "$last" "$demo/screenshot.png"

magick -loop 0 \
    -delay 160 "${frames[0]}" "${frames[1]}" \
    -delay 200 "${frames[2]}" \
    -delay 240 "${frames[3]}" \
    -delay 40 "${frames[@]:4:$((count - 5))}" \
    -delay 480 "$last" \
    -colors 128 -dither None -layers Optimize \
    "$demo/demo.gif"

printf 'built %s (%sx%s, %s frames, %s)\n' \
    "$demo/demo.gif" \
    "$(identify -format '%w' "$demo/demo.gif")" \
    "$(identify -format '%h' "$demo/demo.gif")" \
    "$count" \
    "$(du -h "$demo/demo.gif" | cut -f1)"
