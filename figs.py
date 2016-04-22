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

def fft_plots(psfex, W, H, ps, ps2, ps3):

    H,W = 256,256
    
    psfim = psfex.instantiateAt(0,0)
    mx = psfim.max()
    
    T2 = 15
    psfim = psfim[T2:-T2,T2:-T2]
    ph,pw = psfim.shape
    cx,cy = pw/2, ph/2
    #egal = EllipseE.fromRAbPhi(3., 0.3, 135.)
    egal = EllipseE.fromRAbPhi(4., 0.3, 135.)
    #egal = EllipseE.fromRAbPhi(5., 0.3, 135.)
    gal = ExpGalaxy(PixPos(cx,cy), Flux(100.), egal)
    #gal = GaussianGalaxy(PixPos(cx,cy), Flux(100.), egal)

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
    
    tinyp = gal.getModelPatch(img)
    print 'tiny PSF conv galaxy'
    mod = np.zeros_like(psfim)
    tinyp.addTo(mod)
    tinymod = mod.copy()

    tinypad = np.zeros((pH,pW))
    tinyp.addTo(tinypad)
    
    plt.figure(1)
    
    # unconvolved galaxy image
    plt.clf()
    dimshow(mod, **ima)
    ax = plt.axis()
    for va in amix.var:
        # print 'variance', v
        e = EllipseE.fromCovariance(va)
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
    gal_pix_ax = ax
    
    img.psf = pixpsf
    p = gal.getModelPatch(img)
    print 'PixPSF conv galaxy'
    mod = np.zeros_like(psfim)
    p.addTo(mod)

    # Convolved galaxy image
    plt.clf()
    dimshow(mod, **ima)
    ps.savefig()


    # Fourier-space MoG:
    plt.figure(2)
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fsum.real, Fsum.imag), axes=(0,)), **fima)
    ax = plt.axis()
    for k in range(amix.K):
        print('Pixel-space variance:', amix.var[k,:,:])
        Cinv = np.linalg.inv(amix.var[k,:,:])
        print('Inverse:', Cinv)
        Cinv *= (4. * np.pi**2)
        e = EllipseE.fromCovariance(Cinv)
        B = e.getRaDecBasis()
        B *= 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        H,W = Fsum.real.shape
        plt.plot(xx, H/2. + yy, 'r-', lw=2)
    plt.axis(ax)
    ps.savefig()

    # Just the galaxy MoG in Fourier space, no background image
    plt.clf()
    plt.axhline(H/2., color='k', alpha=0.3)
    plt.axvline(0, color='k', alpha=0.3)
    for k in range(amix.K):
        Cinv = 4. * np.pi**2 * np.linalg.inv(amix.var[k,:,:])
        e = EllipseE.fromCovariance(Cinv)
        B = e.getRaDecBasis() * 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        H,W = Fsum.real.shape
        plt.plot(xx, H/2. + yy, 'r-', lw=2)
    plt.axis(ax)
    plt.xticks([])
    plt.yticks([])
    ps.savefig()
    plt.figure(1)


    # Pixel-space galaxy MoG
    plt.clf()
    plt.axhline(cy, color='k', alpha=0.3)
    plt.axvline(cx, color='k', alpha=0.3)
    for va in amix.var:
        e = EllipseE.fromCovariance(va)
        B = e.getRaDecBasis() * 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        plt.plot(cx + xx, cy + yy, 'r-', lw=2)
    plt.axis(gal_pix_ax)
    plt.xticks([])
    plt.yticks([])
    ps.savefig()


    

    ### Lopass
    plt.figure(2)

    print('w range:', np.min(w), np.max(w))
    print('v range:', np.min(v), np.max(v))
    w2 = np.linspace(2.*np.min(w), 2.*np.max(w), len(w)*2-1)
    v2 = np.linspace(2.*np.min(v), 2.*np.max(v), len(v)*2-1)
    print('w:', w)
    print('w2:', w2)
    print('v:', v)
    print('v2:', v2)
    Fsum2 = amix.getFourierTransform(w2, v2)

    plt.clf()
    dimshow(np.hypot(Fsum2.real, Fsum2.imag), **fima)
    ps3.savefig()

    print('Fsum2.real sum', Fsum2.real.sum())

    print('Fsum.real sum', Fsum.real.sum())

    Ftiny = np.fft.rfft2(tinypad)
    print('Tinypad shape', tinypad.shape)
    tH,tW = tinypad.shape
    Ftiny /= (tH * np.pi)

    
    print('Ftiny.real sum', Ftiny.real.sum())
    
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)), **fima)
    ps3.savefig()

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fsum.real, Fsum.imag), axes=(0,)), **fima)
    ps3.savefig()

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Ftiny.real - Fsum.real,
                                     Ftiny.imag - Fsum.imag), axes=(0,)), **fima)
    ps3.savefig()
    

    plt.figure(1)
    
    w,h = 32,32

    egal = EllipseE.fromRAbPhi(20., 0.1, 115.)
    gal = ExpGalaxy(PixPos(0.,0.), Flux(100.), egal)


    cx,cy = w, h/2
    gal.pos = PixPos(cx, cy)

    p = gal.getModelPatch(img)
    mod = np.zeros((h,w*2))
    p.addTo(mod)

    #halfsize = gal._getUnitFluxPatchSize(img, cx, cy, 0.)
    #print 'halfsize:', halfsize
    # -> 256 x 256 FFT
    
    # Not-wrapped-around galaxy image
    plt.figure(3)
    plt.clf()
    dimshow(mod, **ima)
    ps2.savefig()


    cx,cy = w/2, h/2
    gal.pos = PixPos(cx, cy)

    p = gal.getModelPatch(img, modelMask=Patch(0,0,np.ones((h,w),bool)))
    mod = np.zeros((h,w))
    p.addTo(mod)
    
    # Wrapped-around galaxy image
    plt.figure(1)
    plt.clf()
    dimshow(mod, **ima)
    ps2.savefig()






    


if __name__ == '__main__':
        
    psffn = 'decam-00348226-N18.fits'
    
    W,H = 2048,4096
    psfex = PsfEx(psffn, W, H)
    
    ps = PlotSequence('psf', suffixes=['pdf'])
    ps2 = PlotSequence('gal', suffixes=['pdf'])
    ps3 = PlotSequence('lopass', suffixes=['pdf'])
    
    plt.figure(1, figsize=(3,3))
    margin = 0.01
    #plt.subplots_adjust(left=margin, right=1.-margin, bottom=margin, top=1.-margin,
    #                    hspace=0, wspace=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    #plt.figure(2, figsize=(1.5,3))
    frac = 17./32
    plt.figure(2, figsize=(frac * 3,3))
    # plt.subplots_adjust(left=margin/frac, right=1 - margin/frac,
    #                     bottom=margin, top=1.-margin,
    #                     hspace=0, wspace=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    
    plt.figure(3, figsize=(6,3))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    
    
    plt.figure(1)
    
    fft_plots(psfex, W, H, ps, ps2, ps3)

    
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
