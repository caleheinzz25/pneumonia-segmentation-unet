.PHONY: all clean pdf

DOCNAME = manual_book

all: pdf

pdf: $(DOCNAME).pdf

$(DOCNAME).pdf: $(DOCNAME).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCNAME).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCNAME).tex

clean:
	rm -f *.aux *.log *.out *.pdf *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
