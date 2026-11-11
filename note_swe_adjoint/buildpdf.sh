#!/bin/bash
# buildpdf.sh — compile a LaTeX document with xelatex + bibtex.
# Usage: ./buildpdf.sh file.tex   (extension optional: ./buildpdf.sh file)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 file.tex"
    exit 1
fi

src="$1"
base="${src%.tex}"

if [ ! -f "$base.tex" ]; then
    echo "Error: $base.tex not found"
    exit 1
fi

xelatex "$base.tex"
bibtex "$base.aux"
xelatex "$base.tex"
xelatex "$base.tex"

# Remove auxiliary files for this job only, so other tex projects are untouched
rm -f "$base".aux "$base".dvi "$base".log "$base".toc "$base".bbl "$base".blg "$base".fdb_latexmk "$base".fls "$base".out "$base".fff "$base".lof "$base".log "$base"~ "$base".tex~

evince "$base.pdf" &
