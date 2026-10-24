#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
document_dir="$PWD"
build_dir="$document_dir/build/adjoint"
mkdir -p "$build_dir"
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" adjoint.tex
(
    cd "$build_dir"
    BIBINPUTS="$document_dir:${BIBINPUTS:-}" bibtex adjoint
)
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" adjoint.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" adjoint.tex
cp -- "$build_dir/adjoint.pdf" "$document_dir/adjoint.pdf"
printf 'Built %s\n' "$document_dir/adjoint.pdf"
