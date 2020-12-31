all: gal-conv.pdf

PSF_FIGS := psf-00.pdf psf-01.pdf psf-02.pdf psf-03.pdf psf-05.pdf psf-07.pdf psf-08.pdf
# psf-04.pdf
GAL_FIGS := gal-00.pdf gal-01.pdf gal-02.pdf

MORE_FIGS := lopass-mine-pix.pdf lopass-mine-logfourier.pdf lopass-naive-pix.pdf \
lopass-naive-logfourier.pdf lopass-diff-pixmine-fourier.pdf \
lopass-diff-pixmine-fourier-ann.pdf lopass-diff-pixmine-pix.pdf \
lopass-dpix-pix.pdf lopass-dclip-logfourier.pdf lopass-diff-dclipmine-fourier.pdf \
lopass-diff-dclipmine-fourier-ann.pdf lopass-diff-dclipmine-pix.pdf

gal-conv.pdf: gal-conv.tex $(PSF_FIGS) $(GAL_FIGS)
	pdflatex gal-conv
	pdflatex gal-conv

$(PSF_FIGS) $(GAL_FIGS): figs.py
	python figs.py

arxiv.tgz:
	tar czf $@ gal-conv.tex aastex63.cls $(PSF_FIGS) $(GAL_FIGS) $(MORE_FIGS)
