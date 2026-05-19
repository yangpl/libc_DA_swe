xelatex adjoint.tex
bibtex adjoint.aux
xelatex adjoint.tex
xelatex adjoint.tex
#dvipdfm adjoint.dvi

rm -rf *.aux
rm -rf *.dvi
rm -rf *.log
rm -rf *.toc
rm -rf *.bbl
rm -rf *.blg
rm -rf *.out
rm -rf *.fff
rm -rf *.lof
rm *~

evince adjoint.pdf &
