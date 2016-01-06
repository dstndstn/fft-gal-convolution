all: gal-conv.pdf

PSF_FIGS := psf-00.pdf psf-01.pdf psf-02.pdf psf-03.pdf psf-04.pdf psf-05.pdf
GAL_FIGS := gal-00.pdf gal-01.pdf

gal-conv.pdf: gal-conv.tex $(PSF_FIGS) $(GAL_FIGS)
	pdflatex gal-conv
	pdflatex gal-conv

$(PSF_FIGS) $(GAL_FIGS): figs.py
	python figs.py
