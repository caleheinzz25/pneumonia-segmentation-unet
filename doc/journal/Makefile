.PHONY: all clean pdf docx

# Nama file utama tanpa ekstensi
DOCNAME = journal

all: pdf docx

pdf: $(DOCNAME).pdf

docx: $(DOCNAME).docx

$(DOCNAME).pdf: $(DOCNAME).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCNAME).tex
	biber $(DOCNAME) || true
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCNAME).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCNAME).tex

$(DOCNAME).docx: $(DOCNAME).tex references.bib reference.docx harvard.csl
	pandoc $(DOCNAME).tex -o $(DOCNAME).docx \
		--bibliography=references.bib \
		--csl=harvard.csl \
		--metadata lang=id-ID \
		--reference-doc=reference.docx \
		--citeproc \
		--wrap=none

clean:
	rm -f *.aux *.log *.out *.pdf *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg $(DOCNAME).docx
