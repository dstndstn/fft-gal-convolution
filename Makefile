all: gal-conv.pdf

PSF_FIGS := psf-00.pdf psf-01.pdf psf-02.pdf psf-03.pdf psf-04.pdf psf-05.pdf

gal-conv.pdf: gal-conv.tex $(PSF_FIGS)
	pdflatex gal-conv

$(PSF_FIGS): figs.py
	python figs.py
