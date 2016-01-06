import matplotlib
matplotlib.use('Agg')
import pylab as plt

import tractor
from tractor import *
from tractor.galaxy import *
from astrometry.util.plotutils import *

def plot_psf_ellipses(psffit, dx,dy, *args, **kwargs):
    cx,cy = pw/2, ph/2
    for i,e in enumerate(psffit.ellipses):
        print 'ellipse', e
        B = e.getRaDecBasis()
        B *= 3600.
        print 'basis', B
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        w,mu,var = psffit.get_wmuvar()
        #print 'mean', psffit.mean[i,:]
        print 'mu', mu
        mx,my = mu[i,:] #psffit.mean[i,:]
        plt.plot(cx + xx + mx + dx, cy + yy + my + dy, *args, **kwargs)

def gmm_plots(psfex, W, H, ps):
    psfim = psfex.instantiateAt(W/2, H/2)
    
    T = 10
    psfim = psfim[T:-T, T:-T]
    ph,pw = psfim.shape
    psffit = GaussianMixtureEllipsePSF.fromStamp(psfim, N=2)
    
    mod = np.zeros_like(psfim)
    p = psffit.getPointSourcePatch(pw/2, ph/2, radius=pw/2)
    p.addTo(mod)
    
    mx = psfim.max()
    
    T2 = 5
        
    # # GMM: PSF
    # plt.clf()
    # dimshow(psfim[T2:-T2,T2:-T2], vmin=0, vmax=mx, ticks=False)
    # ps.savefig()
    # 
    # # GMM: PSF + MoG
    # plt.clf()
    # dimshow(mod[T2:-T2,T2:-T2], vmin=0, vmax=mx, ticks=False)
    # ax = plt.axis()
    # plot_psf_ellipses(psffit, -T2,-T2, 'r-', lw=2)
    # plt.axis(ax)
    # ps.savefig()
    # 
    # # GMM: PSF - MoG resid
    # f = 0.05
    # plt.clf()
    # dimshow((psfim - mod)[T2:-T2,T2:-T2], vmin=-f*mx, vmax=f*mx, ticks=False)
    # ps.savefig()
    # 
    # tiny = 1e-20
    # 
    # # GMM: log PSF
    # plt.clf()
    # dimshow(np.log10(np.maximum(psfim,tiny) / mx), vmin=-5, vmax=0, ticks=False)
    # ps.savefig()
    # 
    # # GMM: log MoG
    # plt.clf()
    # dimshow(np.log10(mod / mx), vmin=-5, vmax=0, ticks=False)
    # ps.savefig()
    
    # PSFex spatial grid
    # plt.figure(3, figsize=(7,4))
    # rows,cols = 7,4
    # k = 1
    # for iy,y in enumerate(np.linspace(0, H, rows).astype(int)):
    #     for ix,x in enumerate(np.linspace(0, W, cols).astype(int)):
    #         pim = psfex.instantiateAt(x,y)
    #         pim = pim[15:-15, 15:-15]
    #         mx = pim.max()
    #         kwa = dict(vmin=0, vmax=mx, ticks=False)
    #         plt.subplot(cols, rows, k)
    #         k += 1
    #         dimshow(pim, **kwa)
    # ps.savefig()
    # plt.figure(1)
    
    cx,cy = pw/2, ph/2
    tinypsf = NCircularGaussianPSF([0.01], [1.])
    
    #egal = EllipseESoft(2., 0.2, 0.5)
    # egal EllipseE: re=10, e1=-7.87273e-17, e2=0.428571
    egal = EllipseE.fromRAbPhi(8., 0.3, 135.)
    print 'egal', egal
    gal = ExpGalaxy(PixPos(cx,cy), Flux(100.), egal)
    gal.halfsize = pw/2
    
    data=np.zeros((H,W), np.float32)
    img = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)
    
    tinyp = gal.getModelPatch(img)
    
    amix = gal._getAffineProfile(img, cx, cy)
    #print 'amix', amix
    
    # # GMM: pixelized galaxy
    # plt.clf()
    # dimshow(tinyp.patch)
    # ps.savefig()
    
    # # GMM: galaxy MoG
    # ax = plt.axis()
    # for v in amix.var:
    #     # print 'variance', v
    #     e = EllipseE.fromCovariance(v)
    #     B = e.getRaDecBasis()
    #     # print 'B', B
    #     B *= 3600.
    #     angle = np.linspace(0, 2.*np.pi, 90)
    #     cc = np.cos(angle)
    #     ss = np.sin(angle)
    #     xx = B[0,0] * cc + B[0,1] * ss
    #     yy = B[1,0] * cc + B[1,1] * ss
    #     plt.plot(cx + xx, cy + yy, 'r-', lw=2)
    # plt.axis(ax)
    # ps.savefig()
    
    img.psf = psffit
    p = gal.getModelPatch(img)
    
    # # GMM: galaxy conv PSF
    # plt.clf()
    # dimshow(p.patch)
    # ps.savefig()
    
    psfmix = psffit.getMixtureOfGaussians(px=0, py=0)
    cmix = amix.convolve(psfmix)
    
    # # GMM: galaxy conv PSF MoG
    # ax = plt.axis()
    # for v in cmix.var:
    #     e = EllipseE.fromCovariance(v)
    #     B = e.getRaDecBasis()
    #     B *= 3600.
    #     angle = np.linspace(0, 2.*np.pi, 90)
    #     cc = np.cos(angle)
    #     ss = np.sin(angle)
    #     xx = B[0,0] * cc + B[0,1] * ss
    #     yy = B[1,0] * cc + B[1,1] * ss
    #     plt.plot(cx + xx, cy + yy, 'r-', lw=2)
    # plt.axis(ax)
    # ps.savefig()


##############################################

def fft_plots(psfex, W, H, ps):
    
    psfim = psfex.instantiateAt(W, H)
    mx = psfim.max()
    
    T2 = 15
    psfim = psfim[T2:-T2,T2:-T2]
    ph,pw = psfim.shape
    cx,cy = pw/2, ph/2
    egal = EllipseE.fromRAbPhi(3., 0.3, 135.)
    gal = ExpGalaxy(PixPos(cx,cy), Flux(100.), egal)

    ima = dict(ticks=False, cmap=antigray)
    #fima = dict(ticks=False, cmap='hot_r')
    fima = dict(ticks=False, cmap='Blues')
    
    # PSF img
    plt.clf()
    dimshow(psfim, vmin=0, vmax=mx, **ima)
    ps.savefig()
    
    pixpsf = PixelizedPSF(psfim)#.patch)
    halfsize = 10.
    P,(px0,py0),(pH,pW),(w,v) = pixpsf.getFourierTransform(0., 0., halfsize)

    print 'PSF size:', psfim.shape
    print 'Padded PSF size:', pH,pW
    print 'FFT size:', P.shape, P.dtype
    
    data=np.zeros((H,W), np.float32)
    tinypsf = NCircularGaussianPSF([1e-6], [1.])
    img = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)
    
    #gal.shape.re = 3.
    #print 'galaxy:', gal
    amix = gal._getAffineProfile(img, cx, cy)
    Fsum = amix.getFourierTransform(w, v)
    
    print 'PSF FT'
    plt.figure(2)
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(P.real, P.imag), axes=(0,)), **fima)
    ps.savefig()
    
    # plt.clf()
    # plt.subplot(1,2,1)
    # dimshow(np.fft.fftshift(P.real, axes=(0,)))
    # plt.subplot(1,2,2)
    # dimshow(np.fft.fftshift(P.imag, axes=(0,)))
    # ps.savefig()
    
    print 'Gal FT'
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fsum.real, Fsum.imag), axes=(0,)), **fima)
    ps.savefig()
    
    print 'PSF * Gal FT'
    plt.clf()
    FG = Fsum * P
    dimshow(np.fft.fftshift(np.hypot(FG.real, FG.imag), axes=(0,)), **fima)
    ps.savefig()
    
    plt.figure(1)
    
    tinyp = gal.getModelPatch(img)
    print 'tiny PSF conv galaxy'
    mod = np.zeros_like(psfim)
    tinyp.addTo(mod)
    
    # unconvolved galaxy
    plt.clf()
    dimshow(mod, **ima)
    
    ax = plt.axis()
    for v in amix.var:
        # print 'variance', v
        e = EllipseE.fromCovariance(v)
        B = e.getRaDecBasis()
        # print 'B', B
        B *= 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        plt.plot(cx + xx, cy + yy, 'r-', lw=2)
    plt.axis(ax)
    ps.savefig()
    
    img.psf = pixpsf
    p = gal.getModelPatch(img)
    print 'PixPSF conv galaxy'
    mod = np.zeros_like(psfim)
    p.addTo(mod)

    # Convolved galaxy
    plt.clf()
    dimshow(mod, **ima)
    ps.savefig()

        
psffn = 'decam-00348226-N18.fits'

W,H = 2048,4096
psfex = PsfEx(psffn, W, H)

ps = PlotSequence('psf', suffixes=['pdf'])

plt.figure(1, figsize=(3,3))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                    hspace=0, wspace=0)
plt.figure(2, figsize=(1.5,3))
plt.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99,
                    hspace=0, wspace=0)

plt.figure(1)

fft_plots(psfex, W, H, ps)

    
# plt.clf()
# mx = tinyp.patch.max()
# dimshow(np.log10(tinyp.patch/mx), vmin=-3, vmax=0)
# ps.savefig()

# tpsf = PixelizedPSF(tinyp.patch)
# Ftiny,nil,nil = tpsf.getFourierTransform(25)
# plt.clf()
# dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)))
# ps.savefig()


## egal2 = EllipseE.fromRAbPhi(1., 0.8, 135.)
## gal2 = ExpGalaxy(PixPos(cx,cy), Flux(100.), egal2)
## gal2.halfsize = pw/2
## amix2 = gal2._getAffineProfile(img, cx, cy)
## Fsum2 = amix2.getFourierTransform(w, v)
## 
## plt.clf()
## dimshow(np.fft.fftshift(np.hypot(Fsum2.real, Fsum2.imag), axes=(0,)))
## ps.savefig()
## 
## 
## tinyp = gal2.getModelPatch(img)
## # tpsf = PixelizedPSF(tinyp.patch)
## # Ftiny,nil,nil = tpsf.getFourierTransform(25)
## # plt.clf()
## # dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)))
## # ps.savefig()
## 
## for rad in [0.5, 0.2, 0.1]:
##     egal2 = EllipseE.fromRAbPhi(rad, 0.8, 135.)
##     gal2 = ExpGalaxy(PixPos(cx,cy), Flux(100.), egal2)
##     gal2.halfsize = pw/2
##     amix2 = gal2._getAffineProfile(img, cx, cy)
##     Fsum2 = amix2.getFourierTransform(w, v)
## 
##     print 'radius', rad
##     
##     plt.clf()
##     dimshow(np.fft.fftshift(np.hypot(Fsum2.real, Fsum2.imag), axes=(0,)))
##     ps.savefig()
## 
##     tinyp = gal2.getModelPatch(img)
##     plt.clf()
##     dimshow(tinyp.patch)
##     ps.savefig()
##     plt.clf()
##     mx = tinyp.patch.max()
##     dimshow(np.log10(tinyp.patch/mx), vmin=-3, vmax=0)
##     ps.savefig()
##     mod = np.zeros((25,25))
##     tinyp.addTo(mod)
##     tpsf = PixelizedPSF(mod)
##     Ftiny,nil,nil = tpsf.getFourierTransform(25)
##     plt.clf()
##     dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)))
##     ps.savefig()
