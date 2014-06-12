PAPER = main
SOURCE = ${PAPER}.tex abstract.tex

all: ${PAPER}.pdf

quick: ${SOURCE}
	pdflatex ${PAPER}

skim:
	skimreload ${PAPER}.pdf

${PAPER}.pdf: ${SOURCE}
	pdflatex ${PAPER}.tex
	bibtexall
	pdflatex ${PAPER}.tex
	pdflatex ${PAPER}.tex

clean:
	rm -f ${PAPER}.aux ${PAPER}.blg ${PAPER}.pdf ${PAPER}.bbl ${PAPER}.log ${PAPER}.out

