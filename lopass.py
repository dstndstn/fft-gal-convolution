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

def lopass(ps, fig_square, fig_rect):
    '''
    These plots are for the figure comparing my method to naive
    pixel-space convolution, where the naive (undersampled) one is
    shown to have aliased high-frequency power.
    '''
    ima = dict(ticks=False, cmap=antigray)
    fima = dict(ticks=False, cmap='Blues')

    H = W = 32
    # width of real Fourier transforms
    FW = W//2+1
    w = np.fft.rfftfreq(W)
    v = np.fft.fftfreq(H)

    # The galaxy model
    theta = 110
    egal = EllipseE.fromRAbPhi(4., 0.3, theta)
    gal = ExpGalaxy(PixPos(0,0), Flux(100.), egal)

    # The matching naive galaxy profile rendering function
    # (SDSS) exp model
    naive_func = hogg_exp

    # The matching Gaussian mixture model
    from tractor.mixture_profiles import get_exp_mixture
    expmix = get_exp_mixture()
    gaussian_mixture = expmix

    # While investigating this figure, I tried a few different methods:
    # - pix = naive: evaluate the Galaxy profile in pixel space
    # - mine = Fourier space Gaussian
    # - gpix = evaluate the MoG approximate galaxy profile in pixel space
    # - tiny = render using my method and a tiny Gaussian PSF
    # - dpix = evaluate the galaxy profile at double resolution
    #   - dclip = dpix, but clipped back to normal size
    #
    # "tiny" is basically pointless, but kept for historical interest

    data = np.zeros((H,W), np.float32)
    img = Image(data=data, invvar=np.ones_like(data)) #, psf=tinypsf)
    
    # Compute our galaxy FFT
    amix = gal._getAffineProfile(img, 0, 0)
    Fmine = amix.getFourierTransform(w, v)
    mine = np.fft.fftshift(np.fft.irfft2(Fmine, s=(H,W)))
    mx = mine.max()

    # "pix" = "naive": galaxy profile rendered at pixel centers
    xx,yy = np.meshgrid(np.arange(0, H), np.arange(0, H))
    midx = W//2
    midy = H//2
    dx,dy = xx - midx, yy - midy
    # -get the matrix that takes pixels to r_e coordinates
    cd = np.eye(2) / 3600.
    Tinv = egal.getTensor(cd)
    re_coords = Tinv.dot([dx.ravel(), dy.ravel()])
    re_x,re_y = re_coords[0,:], re_coords[1,:]
    re = np.hypot(re_x, re_y)
    re = re.reshape(H, W)
    pix = naive_func(re)
    pix /= pix.sum()
    Fpix = np.fft.rfft2(np.fft.fftshift(pix))

    # "gpix" = evaluate our Gaussian Mixture Model approximation to the exp profile.
    # (notice that this re-uses "re_coords" from "pix")
    gpix = gaussian_mixture.evaluate(re_coords.T)
    gpix = gpix.reshape(H, W)
    gpix /= gpix.sum()
    Fgpix = np.fft.rfft2(np.fft.fftshift(gpix))
    
    # "dpix" = rendered at double resolution
    xx,yy = np.meshgrid(np.arange(0, W, 0.5), np.arange(0, H, 0.5))
    dx,dy = xx - midx, yy - midy
    re_coords = Tinv.dot([dx.ravel(), dy.ravel()])
    re_x,re_y = re_coords[0,:], re_coords[1,:]
    re = np.hypot(re_x, re_y)
    re = re.reshape(H*2,W*2)
    dpix = naive_func(re)
    dpix /= dpix.sum()
    Fdpix = np.fft.rfft2(np.fft.fftshift(dpix))

    dh,dw = Fdpix.shape
    # "dclip" = dpix cut to the central frequency region
    print('Double shape:', dh,dw)
    print('Normal shape:', H, W)
    clip = (dh - H)//2
    shifted = np.fft.fftshift(Fdpix, axes=(0,))
    shifted = shifted[clip:-clip, :-clip]
    print('clipped shape:', shifted.shape)
    Fdclip = np.fft.fftshift(shifted, axes=(0,))
    assert(Fdclip.shape == (H,FW))

    
    
    if False:
        # "tiny": render with a tiny PSF
        # Move galaxy to center of image.
        gal.pos.x = H/2.
        gal.pos.y = W/2.
        tinypsf = NCircularGaussianPSF([1e-6], [1.])
        tiny = np.zeros((H,W))
        gal.getModelPatch(img).addTo(tiny)
        Ftiny = np.fft.rfft2(np.fft.fftshift(tiny))
        Ftiny /= (H * np.pi)
        plt.clf()
        dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
                vmin=0, vmax=1.1, **fima)
        plt.title('Fourier space, tiny psf')
        ps.savefig()
        plt.clf()
        dimshow(tiny, vmin=0, vmax=mx, **ima)
        plt.title('Tiny psf')
        ps.savefig()
        plt.clf()
        dimshow(np.log10(np.maximum(
            np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
            1e-6)), vmin=-3, vmax=0, **fima)
        plt.title('log fourier: tiny')
        ps.savefig()
        plt.clf()
        dimshow(np.fft.fftshift(np.hypot(Ftiny.real - Fmine.real,
                                         Ftiny.imag - Fmine.imag), axes=(0,)), **fima)
        plt.title('Fourier: tiny - mine')
        ps.savefig()

    def diffshow(D, **ima):
        mx = np.max(np.abs(D))
        dimshow(D, vmin=-mx, vmax=mx, **ima)
    
    
    plt.clf()
    dimshow(mine, vmin=0, vmax=mx, **ima)
    plt.title('Mine')
    ps.savefig()

    plt.clf()
    dimshow(pix, vmin=0, vmax=mx, **ima)
    plt.title('Direct pixelized')
    ps.savefig()

    plt.clf()
    dimshow(gpix, **ima)
    plt.title('Pixelized Gaussians (gpix)')
    ps.savefig()

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fpix.real, Fpix.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    plt.title('Fourier pixelized')
    ps.savefig()

    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fpix.real, Fpix.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log fourier: pix')
    ps.savefig()

    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fgpix.real, Fgpix.imag), axes=(0,)),
        1e-6)), vmin=-3, vmax=0, **fima)
    plt.title('log fourier: gpix')
    ps.savefig()

    # Log Fourier plots: mine
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fmine.real, Fmine.imag), axes=(0,)),
        1e-6)), vmin=-3, vmax=0, **fima)
    plt.title('log fourier: mine')
    ps.savefig()

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fpix.real - Fmine.real,
                                     Fpix.imag - Fmine.imag), axes=(0,)), **fima)
    plt.title('Fourier: pix - mine')
    ps.savefig()

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fgpix.real - Fmine.real,
                                     Fgpix.imag - Fmine.imag), axes=(0,)), **fima)
    plt.title('Fourier: gpix - mine')
    ps.savefig()

    diff = np.fft.fftshift(np.fft.irfft2(Fpix - Fmine, s=pix.shape))
    plt.clf()
    dimshow(diff, **ima)
    plt.title('IFFT pix - mine')
    ps.savefig()    

    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(
            np.hypot(Fdpix.real, Fdpix.imag),
            axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log Fourier: double-res pix')
    ps.savefig()

    ax = plt.axis()
    for k in range(1, amix.K):
        Cinv = np.linalg.inv(amix.var[k,:,:])
        Cinv *= (4. * np.pi**2)
        e = EllipseE.fromCovariance(Cinv)
        B = e.getRaDecBasis() * 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        plt.plot(xx,  0.5 * dh + yy, 'r-', lw=2)
        plt.plot(xx,  1.5 * dh + yy, 'r--', lw=2)
        plt.plot(xx, -0.5 * dh + yy, 'r--', lw=2)
    clip = (dh - H)//2
    plt.plot([0, FW, FW, 0], [clip, clip, dh-clip, dh-clip], 'k--')

    plt.axis(ax)
    ps.savefig()
    
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(
            np.hypot(Fdclip.real, Fdclip.imag),
            axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    plt.title('log Fourier: clipped double-res pix')
    ps.savefig()

    ax = plt.axis()
    for k in range(1, amix.K):
        Cinv = np.linalg.inv(amix.var[k,:,:])
        Cinv *= (4. * np.pi**2)
        e = EllipseE.fromCovariance(Cinv)
        B = e.getRaDecBasis() * 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        plt.plot(xx, 0.5*H + 0*dh + yy, 'r-',  lw=2)
        plt.plot(xx, 0.5*H + 1*dh + yy, 'r--', lw=2)
        plt.plot(xx, 0.5*H - 1*dh + yy, 'r--', lw=2)
    plt.axis(ax)
    ps.savefig()

    diff = np.hypot(Fdclip.real - Fmine.real,
                    Fdclip.imag - Fmine.imag)
    print('Diff max:', diff.max())
    mx = diff.max()
    
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fdclip.real - Fmine.real,
                                     Fdclip.imag - Fmine.imag), axes=(0,)),
                                     vmin=0, vmax=mx/2, **fima)
    plt.title('Fourier: double - mine')
    ps.savefig()

    diff = np.fft.fftshift(np.fft.irfft2(Fdclip - Fmine, s=pix.shape))
    plt.clf()
    dimshow(diff, **ima)
    plt.title('IFFT double - mine')
    ps.savefig()    
    


if __name__ == '__main__':
    disable_galaxy_cache()
    ps = PlotSequence('lopass2')

    fig_square = fig_rect = 1

    #plt.figure()
    #plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    lopass(ps, fig_square, fig_rect)
    sys.exit(0)
    
