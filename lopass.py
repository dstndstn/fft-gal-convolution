from __future__ import print_function
import sys
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import tractor
from tractor import *
from tractor.galaxy import *
from astrometry.util.plotutils import *

#### From hogg's TheTractor/optimize_mixture_profiles.py
from scipy.special import gammaincinv
def sernorm(n):
	return gammaincinv(2.*n, 0.5)
def hogg_ser(x, n, soft=None):
    if soft:
        return np.exp(-sernorm(n) * ((x**2 + soft) ** (1./(2.*n)) - 1.))
    return np.exp(-sernorm(n) * (x ** (1./n) - 1.))
def hogg_exp(x):
    """
    One-dimensional exponential profile.

    Normalized to return 1. at x=1.
    """
    return hogg_ser(x, 1.)
####



def lopass(psfex, ps3):
    '''
    These plots are for the figure comparing my method to naive
    pixel-space convolution, where the naive (undersampled) one is
    shown to have aliased high-frequency power.
    '''

    H,W = 256,256

    # The PSF image -- we'll grab out a central stamp
    psfim = psfex.instantiateAt(0,0)
    sub = 15
    psfim = psfim[sub:-sub,sub:-sub]
    ph,pw = psfim.shape
    cx,cy = pw//2, ph//2
    pixpsf = PixelizedPSF(psfim)
    halfsize = 10.
    P,(px0,py0),(pH,pW),(w,v) = pixpsf.getFourierTransform(0., 0., halfsize)

    # The galaxy model
    theta = 110
    egal = EllipseE.fromRAbPhi(4., 0.3, theta)
    gal = ExpGalaxy(PixPos(0,0), Flux(100.), egal)

    # Instead of actually rendering the naive model, we fake it using a
    # PSF with a tiny width.
    data = np.zeros((H,W), np.float32)
    tinypsf = NCircularGaussianPSF([1e-6], [1.])
    img = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)

    # Compute our galaxy FFT
    amix = gal._getAffineProfile(img, 0, 0)
    Fmine = amix.getFourierTransform(w, v)

    ima = dict(ticks=False, cmap=antigray)
    fima = dict(ticks=False, cmap='Blues')
    rima = dict(ticks=False, cmap='Greens')
    iima = dict(ticks=False, cmap='Reds')

    def diffshow(D, **ima):
        mx = np.max(np.abs(D))
        dimshow(D, vmin=-mx, vmax=mx, **ima)
    
    # Move galaxy to center of image.
    gal.pos.x = pH/2.
    gal.pos.y = pW/2.

    tinyp = gal.getModelPatch(img)
    print('Tinypatch:', tinyp.shape)
    tinypad = np.zeros((pH,pW))
    tinyp.addTo(tinypad)

    # Galaxy conv tiny PSF
    # Rotated to be zero-centered.
    tinypad2 = np.fft.fftshift(tinypad)
    Ftiny = np.fft.rfft2(tinypad2)
    tH,tW = tinypad.shape
    Ftiny /= (tH * np.pi)

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    plt.title('Fourier space, tiny psf')
    ps3.savefig()

    PM = np.fft.irfft2(Fmine, s=(pH,pW))
    PT = np.fft.irfft2(Ftiny, s=(tH,tW))
    mx = max(PM.max(), PT.max())
    PM = np.fft.fftshift(PM)
    PT = np.fft.fftshift(PT)

    plt.clf()
    dimshow(PM, vmin=0, vmax=mx, **ima)
    plt.title('Mine')
    ps3.savefig()
    
    plt.clf()
    dimshow(PT, vmin=0, vmax=mx, **ima)
    plt.title('Tiny psf')
    ps3.savefig()

    hh,ww = PT.shape
    # Rendered at pixel centers
    xx,yy = np.meshgrid(np.arange(0, ww), np.arange(0, hh))
    midx = ww//2
    midy = hh//2
    dx,dy = xx - midx, yy - midy
    # -get the matrix that takes pixels to r_e coordinates
    cd = np.eye(2) / 3600.
    Tinv = egal.getTensor(cd)
    re_coords = Tinv.dot([dx.ravel(), dy.ravel()])
    re_x,re_y = re_coords[0,:], re_coords[1,:]
    re = np.hypot(re_x, re_y)
    re = re.reshape(hh, ww)
    # Evaluate the (SDSS) exp model
    pix = hogg_exp(re)
    pix /= pix.sum()
    
    plt.clf()
    dimshow(pix, vmin=0, vmax=mx, **ima)
    plt.title('Direct pixelized')
    ps3.savefig()

    # Also evaluate our Gaussian Mixture Model approximation to the exp profile.
    from tractor.mixture_profiles import get_exp_mixture
    expmix = get_exp_mixture()
    gpix = expmix.evaluate(re_coords.T)
    gpix = gpix.reshape(hh,ww)
    gpix /= gpix.sum()
    Fgpix = np.fft.rfft2(np.fft.fftshift(gpix))

    plt.clf()
    dimshow(gpix, **ima)
    plt.title('Rendered profile (gpix)')
    ps3.savefig()

    print('Fourier transform of exp')
    Fexp = np.fft.rfft2(np.fft.fftshift(pix))
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fexp.real, Fexp.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    ps3.savefig()

    print('Log Fourier transform of exp')
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fexp.real, Fexp.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log fourier: pix')
    ps3.savefig()

    # Log Fourier
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log fourier: tiny')
    ps3.savefig()

    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fgpix.real, Fgpix.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log fourier: gpix')
    ps3.savefig()

    # Log Fourier plots: mine
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fmine.real, Fmine.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log fourier: me')
    ps3.savefig()
    
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fexp.real - Fmine.real,
                                     Fexp.imag - Fmine.imag), axes=(0,)), **fima)
    plt.title('Fourier: pix - me')
    ps3.savefig()

    return
    
    # Ftiny2 = np.fft.fft2(tinypad2)
    # Ftiny2 /= (tH * np.pi)
    # print('Ftiny2.real sum', Ftiny2.real.sum())
    # 
    # plt.clf()
    # dimshow(np.fft.fftshift(np.hypot(Ftiny2.real, Ftiny2.imag)),
    #         **fima)
    # plt.colorbar()
    # ps3.savefig()

    # plt.clf()
    # dimshow(np.fft.fftshift(Ftiny.real, axes=(0,)), **rima)
    # ps3.savefig()
    # plt.clf()
    # dimshow(np.fft.fftshift(Ftiny.imag, axes=(0,)), **iima)
    # ps3.savefig()
    print('Ftiny Real range:', Ftiny.real.min(), Ftiny.real.max())
    print('Ftiny Imag range:', Ftiny.imag.min(), Ftiny.imag.max())

    print('Fmine Real range:', Fmine.real.min(), Fmine.real.max())
    print('Fmine Imag range:', Fmine.imag.min(), Fmine.imag.max())
    
    # Mine, Fourier space intensity
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fmine.real, Fmine.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    plt.savefig('lopass-mine.pdf')

    # Log Fourier plots: naive
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.savefig('lopass-naive-log.pdf')
    
    # Pixel-space galaxy plots
    plt.figure(fig_square)
    PM = np.fft.irfft2(Fmine, s=(pH,pW))
    PT = np.fft.irfft2(Ftiny, s=(tH,tW))
    mx = max(PM.max(), PT.max())
    PM = np.fft.fftshift(PM)
    PT = np.fft.fftshift(PT)

    plt.clf()
    dimshow(PM, vmin=0, vmax=mx, **ima)
    plt.savefig('lopass-mine-pix.pdf')
    
    plt.clf()
    dimshow(PT, vmin=0, vmax=mx, **ima)
    plt.savefig('lopass-naive-pix.pdf')

    print('PT shape', PT.shape)
    print('PM shape', PM.shape)
    
    # Also try rendering the galaxy in pixel space using the real profile.

    # -get the matrix that takes pixels to r_e coordinates
    cd = np.eye(2) / 3600.
    Tinv = egal.getTensor(cd)
    hh,ww = PM.shape
    xx,yy = np.meshgrid(np.arange(ww), np.arange(hh))
    midx,midy = ww//2, hh//2
    dx,dy = xx - midx, yy - midy
    re_coords = Tinv.dot([dx.ravel(), dy.ravel()])
    # Convert re_x, re_y to just radial r_e
    re_x,re_y = re_coords[0,:], re_coords[1,:]
    re = np.hypot(re_x, re_y)
    re = re.reshape(hh,ww)
    # Evaluate the (SDSS) exp model
    pix = hogg_exp(re)

    print('Rendered exp')
    plt.clf()
    dimshow(pix, **ima)
    plt.title('Rendered profile (exp)')
    ps3.savefig()
    #measure(pix)

    print('Pix mine')
    plt.clf()
    dimshow(PM, **ima)
    ps3.savefig()

    d = PT/PT.sum() - pix/pix.sum()
    print('Relative diff exp-Naive:', d.min(), d.max())

    # Also evaluate our Gaussian Mixture Model approximation to the exp profile.
    from tractor.mixture_profiles import get_exp_mixture
    expmix = get_exp_mixture()
    gpix = expmix.evaluate(re_coords.T)
    gpix = gpix.reshape(hh,ww)

    gd = PT/PT.sum() - gpix/gpix.sum()
    print('Relative diff gmix-Naive:', gd.min(), gd.max())
    
    print('Rendered Gaussian mixture')
    plt.clf()
    dimshow(gpix, **ima)
    plt.title('Rendered profile (gmix)')
    ps3.savefig()
    #measure(gpix)

    # plt.clf()
    # diffshow(PT/PT.sum() - pix/pix.sum(), cmap='RdBu')
    # ps3.savefig()

    #print('Sum PM:', PM.sum())
    pix /= pix.sum()
    
    print('Fourier transform of exp')
    Fexp = np.fft.rfft2(np.fft.fftshift(pix))
    # escale = np.sum(np.hypot(Ftiny.real, Ftiny.imag)) / np.sum(np.hypot(Fexp.real, Fexp.imag))
    # print('exp scaling:', escale)
    # Fexp *= escale
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fexp.real, Fexp.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    ps3.savefig()

    print('Log Fourier transform of exp')
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fexp.real, Fexp.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    ps3.savefig()

    print('Log Fourier transform of mine')
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fmine.real, Fmine.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    ps3.savefig()
    
    print('Mine real', Fmine.real.min(), Fmine.real.max(),
          'imag', Fmine.imag.min(), Fmine.imag.max())
    plt.clf()
    plt.subplot(1,2,1)
    dimshow(np.fft.fftshift(Fmine.real, axes=(0,)))
    plt.subplot(1,2,2)
    dimshow(np.fft.fftshift(Fmine.imag, axes=(0,)))
    ps3.savefig()

    print('Exp real', Fexp.real.min(), Fexp.real.max(),
          'imag', Fexp.imag.min(), Fexp.imag.max())
    plt.clf()
    plt.subplot(1,2,1)
    dimshow(np.fft.fftshift(Fexp.real, axes=(0,)))
    plt.subplot(1,2,2)
    dimshow(np.fft.fftshift(Fexp.imag, axes=(0,)))
    ps3.savefig()

    # print('Tiny real', Ftiny.real.min(), Ftiny.real.max(),
    #       'imag', Ftiny.imag.min(), Ftiny.imag.max())
    # plt.clf()
    # plt.subplot(1,2,1)
    # dimshow(np.fft.fftshift(Ftiny.real, axes=(0,)))
    # plt.subplot(1,2,2)
    # dimshow(np.fft.fftshift(Ftiny.imag, axes=(0,)))
    # ps3.savefig()
    
    print('Diff: exp - mine')
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fexp.real - Fmine.real,
                                     Fexp.imag - Fmine.imag), axes=(0,)), **fima)
    ps3.savefig()

    # Rendered at double resolution
    # --- exactly where do we want to put the sub-pixel positions?
    xx,yy = np.meshgrid(np.arange(0, ww, 0.5), np.arange(0, hh, 0.5))
    #xx -= 0.25
    #yy -= 0.25
    dx,dy = xx - midx, yy - midy
    re_coords = Tinv.dot([dx.ravel(), dy.ravel()])
    re_x,re_y = re_coords[0,:], re_coords[1,:]
    re = np.hypot(re_x, re_y)
    re = re.reshape(hh*2,ww*2)
    # Evaluate the (SDSS) exp model
    pix = hogg_exp(re)
    pix /= pix.sum()

    gpix = expmix.evaluate(re_coords.T)
    gpix = gpix.reshape(hh*2,ww*2)
    gpix /= gpix.sum()

    print('Rendered exp (double res)')
    plt.clf()
    dimshow(pix, **ima)
    ps3.savefig()

    print('Fourier transform of exp (double)')
    Fexp = np.fft.rfft2(np.fft.fftshift(pix))
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fexp.real, Fexp.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    ps3.savefig()

    # print('Log Fourier transform of exp (double)')
    # plt.clf()
    # dimshow(np.log10(np.maximum(
    #     np.hypot(Fexp.real, Fexp.imag),
    #     1e-6)),
    #     vmin=-3, vmax=0, **fima)
    # ps3.savefig()
    
    print('Log Fourier transform of exp (double)')
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fexp.real, Fexp.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    ps3.savefig()

    iy,ix = np.unravel_index(np.argmax(np.fft.fftshift(Fmine.real, axes=(0,))),
                             Fmine.real.shape)
    print('Fmine argmax x,y', ix,iy)
    fh,fw = Fmine.real.shape
    print('shape', fh,fw)
    
    fy,fx = np.unravel_index(np.argmax(np.fft.fftshift(Fexp.real, axes=(0,))),
                             Fexp.real.shape)
    print('Fexp argmax x,y', fx,fy)
    print('shape', Fexp.real.shape)

    Fsub = np.fft.fftshift(Fexp, axes=(0,))
    Fsub = Fsub[fy - iy: fy - iy + fh, :fw]

    Fgsub = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(gpix)), axes=(0,))
    Fgsub = Fgsub[fy - iy: fy - iy + fh, :fw]
    
    print('Log Fourier transform of exp (double, sub)')
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.hypot(Fsub.real, Fsub.imag),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    ps3.savefig()

    print('Log Fourier transform of gexp (double, sub)')
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.hypot(Fgsub.real, Fgsub.imag),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    ps3.savefig()

    print('exp(double,sub) - gexp(double,sub)')
    plt.clf()
    dimshow(Fsub.real - Fgsub.real, **fima)
    plt.colorbar()
    ps3.savefig()




if __name__ == '__main__':
    disable_galaxy_cache()
    psffn = 'decam-00348226-N18.fits'
    
    W,H = 2048,4096
    psfex = PsfEx(psffn, W, H)
    ps3 = PlotSequence('lopass2')

    plt.figure()
    #plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    lopass(psfex, ps3)
    sys.exit(0)
    
