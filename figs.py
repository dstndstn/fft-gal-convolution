from __future__ import print_function
import sys
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
        print('ellipse', e)
        B = e.getRaDecBasis()
        B *= 3600.
        print('basis', B)
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        w,mu,var = psffit.get_wmuvar()
        #print('mean', psffit.mean[i,:])
        print('mu', mu)
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
    print('egal', egal)
    gal = ExpGalaxy(PixPos(cx,cy), Flux(100.), egal)
    gal.halfsize = pw/2
    
    data=np.zeros((H,W), np.float32)
    img = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)
    
    tinyp = gal.getModelPatch(img)
    
    amix = gal._getAffineProfile(img, cx, cy)
    #print('amix', amix)
    
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

    print('PSF size:', psfim.shape)
    print('Padded PSF size:', pH,pW)
    print('FFT size:', P.shape, P.dtype)
    
    data=np.zeros((H,W), np.float32)
    tinypsf = NCircularGaussianPSF([1e-6], [1.])
    img = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)
    
    #gal.shape.re = 3.
    #print 'galaxy:', gal
    amix = gal._getAffineProfile(img, cx, cy)
    Fsum = amix.getFourierTransform(w, v)

    print('PSF FT')
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
    
    print('Gal FT')
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fsum.real, Fsum.imag), axes=(0,)), **fima)
    ps.savefig()
    
    print('PSF * Gal FT')
    plt.clf()
    FG = Fsum * P
    dimshow(np.fft.fftshift(np.hypot(FG.real, FG.imag), axes=(0,)), **fima)
    ps.savefig()
    
    tinyp = gal.getModelPatch(img)
    print('tiny PSF conv galaxy')
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
        e = EllipseE.fromCovariance(va)
        B = e.getRaDecBasis()
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
    print('PixPSF conv galaxy')
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

    # gal-*.pdf plots (pixel-space wrap-around)
    
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



    
def lopass(psfex, ps3):
    ### Lopass
    #plt.figure(2)

    fig_rect = 3
    fig_square = 2
    
    H,W = 256,256


    theta = 110
    psfim = psfex.instantiateAt(0,0)

    T2 = 15
    psfim = psfim[T2:-T2,T2:-T2]
    ph,pw = psfim.shape
    cx,cy = pw/2, ph/2
    
    egal = EllipseE.fromRAbPhi(4., 0.3, theta)
    gal = ExpGalaxy(PixPos(0,0), Flux(100.), egal)

    pixpsf = PixelizedPSF(psfim)
    halfsize = 10.
    P,(px0,py0),(pH,pW),(w,v) = pixpsf.getFourierTransform(0., 0., halfsize)

    data=np.zeros((H,W), np.float32)
    tinypsf = NCircularGaussianPSF([1e-6], [1.])
    img = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)

    #amix = gal._getAffineProfile(img, cx, cy)
    amix = gal._getAffineProfile(img, 0, 0)
    Fmine = amix.getFourierTransform(w, v)

    print('Amix amps:', amix.amp, 'sum', amix.amp.sum())

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
    print('tiny PSF conv galaxy')
    print('pH,pW:', pH,pW)
    tinyp = gal.getModelPatch(img)
    tinypad = np.zeros((pH,pW))
    tinyp.addTo(tinypad)

    # #print('w range:', np.min(w), np.max(w))
    # #print('v range:', np.min(v), np.max(v))
    # w2 = np.linspace(2.*np.min(w), 2.*np.max(w), len(w)*2-1)
    # v2 = np.linspace(2.*np.min(v), 2.*np.max(v), len(v)*2-1)
    # # print('w:', w)
    # # print('w2:', w2)
    # # print('v:', v)
    # # print('v2:', v2)
    # # [v2,w2]
    # Fmine2 = amix.getFourierTransform(w2, v2)
    # 
    # # print('w2', len(w2), 'v2', len(v2), 'Fmine2', Fmine2.shape)
    # 
    # I = np.flatnonzero((w2 >= np.min(w)) * (w2 <= np.max(w)))
    # J = np.flatnonzero((v2 >= np.min(v)) * (v2 <= np.max(v)))
    # # print('w', len(w), 'v', len(v), 'Fmine', Fmine.shape)
    # # print('I', len(I), 'J', len(J))
    # 
    # print('Sub-sum Fmine2:', Fmine2[J,:][:,I].real.sum())
    # 
    # # w3 = np.linspace(3.*np.min(w), 3.*np.max(w), len(w)*3-2)
    # # v3 = np.linspace(3.*np.min(v), 3.*np.max(v), len(v)*3-2)
    # # print('w:', w)
    # # print('w3:', w3)
    # # print('v:', v)
    # # print('v3:', v3)
    # # # [v2,w2]
    # # Fmine3 = amix.getFourierTransform(w3, v3)
    # # print('Fmine3.real sum', Fmine3.real.sum())
    # # 
    # # print('Folded Fmine3.real sum', Fmine3.real.sum() + Fmine3[:,1:].real.sum())
    # 
    # #print('amix:', amix)
    # #print('amix means:', amix.mean)
    # 
    # # My method, Fourier transform with twice the frequency range
    # plt.clf()
    # dimshow(np.hypot(Fmine2.real, Fmine2.imag), **fima)
    # plt.title('Fmine2')
    # ps3.savefig()
    # 
    # #for va in amix.var:
    # #    e = EllipseE.fromCovariance(va)
    # ax = plt.axis()
    # for k in range(amix.K):
    #     Cinv = np.linalg.inv(amix.var[k,:,:])
    #     Cinv *= (4. * np.pi**2)
    #     e = EllipseE.fromCovariance(Cinv)
    #     B = e.getRaDecBasis() * 3600.
    #     angle = np.linspace(0, 2.*np.pi, 90)
    #     cc = np.cos(angle)
    #     ss = np.sin(angle)
    #     xx = B[0,0] * cc + B[0,1] * ss
    #     yy = B[1,0] * cc + B[1,1] * ss
    #     f2H,f2W = Fmine2.shape
    #     plt.plot(xx, f2H/2. + yy, 'r-', lw=2)
    # 
    #     plt.plot(xx, 1.5 * f2H + yy, 'r--', lw=2)
    #     plt.plot(xx, -0.5 * f2H + yy, 'r--', lw=2)
    # 
    # plt.axis(ax)
    # ps3.savefig()
    # 
    # # plt.clf()
    # # dimshow(Fmine2.real, **rima)
    # # print('Real range:', Fmine2.real.min(), Fmine2.real.max())
    # # ps3.savefig()
    # # plt.clf()
    # # dimshow(Fmine2.imag, **iima)
    # # print('Imag range:', Fmine2.imag.min(), Fmine2.imag.max())
    # # ps3.savefig()
    # 
    # print('Fmine2.real sum', Fmine2.real.sum())
    # print('Fmine.real sum', Fmine.real.sum())

    # plt.clf()
    # dimshow(tinypad, **ima)
    # ps3.savefig()
    # 
    # Rotated to be zero-centered.
    tinypad2 = np.fft.fftshift(tinypad)
    # plt.clf()
    # dimshow(tinypad2, **ima)
    # ps3.savefig()
    # 
    # my = np.fft.irfft2(Fmine, s=(pH,pW))
    # plt.clf()
    # dimshow(my, **ima)
    # ps3.savefig()

    # Galaxy conv tiny PSF
    #Ftiny = np.fft.rfft2(tinypad)
    Ftiny = np.fft.rfft2(tinypad2)
    #print('Tinypad shape', tinypad.shape)
    tH,tW = tinypad.shape
    Ftiny /= (tH * np.pi)

    print('Ftiny.real sum', Ftiny.real.sum())

    #print('Folded Ftiny.real sum', Ftiny.real.sum() + Ftiny[:,1:].real.sum())

    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    #plt.colorbar()
    #ps3.savefig()
    plt.savefig('lopass-naive.pdf')
    
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

    # Mine, at regular frequencies
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fmine.real, Fmine.imag), axes=(0,)),
            vmin=0, vmax=1.1, **fima)
    #plt.colorbar()
    plt.savefig('lopass-mine.pdf')

    plt.figure(fig_square)
    # Pixel-space galaxy plots
    PM = np.fft.irfft2(Fmine, s=(pH,pW))
    PT = np.fft.irfft2(Ftiny, s=(tH,tW))
    mx = max(PM.max(), PT.max())
    print('PM sum', PM.sum())
    print('Max', PM.max())
    PM = np.fft.fftshift(PM)
    plt.clf()
    dimshow(PM, vmin=0, vmax=mx, **ima)
    plt.savefig('lopass-mine-pix.pdf')
    #ps3.savefig()
    
    print('PT sum', PT.sum())
    PT = np.fft.fftshift(PT)
    plt.clf()
    dimshow(PT, vmin=0, vmax=mx, **ima)
    #ps3.savefig()
    plt.savefig('lopass-naive-pix.pdf')
    plt.figure(fig_rect)

    
    # plt.clf()
    # dimshow(np.fft.fftshift(Fmine.real, axes=(0,)), **rima)
    # ps3.savefig()
    # 
    # plt.clf()
    # dimshow(np.fft.fftshift(Fmine.imag, axes=(0,)), **iima)
    # ps3.savefig()

    print('Fmine Real range:', Fmine.real.min(), Fmine.real.max())
    print('Fmine Imag range:', Fmine.imag.min(), Fmine.imag.max())


    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Ftiny.real - Fmine.real,
                                     Ftiny.imag - Fmine.imag), axes=(0,)), **fima)
    #plt.colorbar()
    plt.savefig('lopass-diff.pdf')


    # plt.clf()
    # dimshow(np.fft.fftshift(Ftiny.real - Fmine.real, axes=(0,)), **rima)
    # ps3.savefig()
    # 
    # plt.clf()
    # dimshow(np.fft.fftshift(Ftiny.imag - Fmine.imag, axes=(0,)), **iima)
    # ps3.savefig()

    diff = Ftiny - Fmine
    print('diff Real range:', diff.real.min(), diff.real.max())
    print('diff Imag range:', diff.imag.min(), diff.imag.max())

    print('Fmine sum:', np.hypot(Fmine.real, Fmine.imag).sum())
    print('Ftiny sum:', np.hypot(Ftiny.real, Ftiny.imag).sum())

    ax = plt.axis()
    ## HACK -- 5th contour is confusing-looking
    for k in range(2, amix.K):
        Cinv = np.linalg.inv(amix.var[k,:,:])
        Cinv *= (4. * np.pi**2)
        e = EllipseE.fromCovariance(Cinv)
        B = e.getRaDecBasis() * 3600.
        angle = np.linspace(0, 2.*np.pi, 90)
        cc = np.cos(angle)
        ss = np.sin(angle)
        xx = B[0,0] * cc + B[0,1] * ss
        yy = B[1,0] * cc + B[1,1] * ss
        fsH,fsW = Fmine.shape
        plt.plot(xx, fsH/2. + yy, 'r-', lw=2)

        plt.plot(xx, 1.5 * fsH + yy, 'r--', lw=2)
        plt.plot(xx, -0.5 * fsH + yy, 'r--', lw=2)
    plt.axis(ax)
    plt.savefig('lopass-diff-ell.pdf')
    #ps3.savefig()


    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Ftiny.real, Ftiny.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    #plt.colorbar()
    #plt.title('log Ftiny')
    #ps3.savefig()
    plt.savefig('lopass-naive-log.pdf')
    
    plt.clf()
    dimshow(np.log10(np.maximum(
        np.fft.fftshift(np.hypot(Fmine.real, Fmine.imag), axes=(0,)),
        1e-6)),
        vmin=-3, vmax=0, **fima)
    #plt.colorbar()
    #plt.title('log Fmine')
    #ps3.savefig()
    plt.savefig('lopass-mine-log.pdf')

    # plt.clf()
    # dimshow(np.log10(np.maximum(
    #     np.hypot(Fmine2.real, Fmine2.imag),
    #     1e-6)),
    #     vmin=-3, vmax=0, **fima)
    # plt.colorbar()
    # plt.title('log Fmine2')
    # ps3.savefig()
        


    # Subsample the PSF via resampling
    from astrometry.util.util import lanczos_shift_image


    # dxlist,dylist = [],[]
    # scales = [2,4,8,16]
    # for scale in scales:
    #     
    #     sh,sw = ph*scale, pw*scale
    #     subpsfim = np.zeros((sh,sw))
    #     step = 1./float(scale)
    #     for ix in np.arange(scale):
    #         for iy in np.arange(scale):
    #             dx = ix * step
    #             dy = iy * step
    #             #dx = 0.5 + (ix - 0.5) * step
    #             #dy = 0.5 + (iy - 0.5) * step
    #             #if ix == 0 and iy == 0:
    #             #    subpsfim[0::scale, 0::scale] = psfim
    #             #    continue
    #             shift = lanczos_shift_image(psfim, -dx, -dy, order=5)
    #             subpsfim[iy::scale, ix::scale] = shift
    # 
    #     # HACK -clamp edges to zero to counter possible weird lanczos edge issues?
    #     subpsfim[:2,:] = 0
    #     subpsfim[:,:2] = 0
    #     subpsfim[-2:,:] = 0
    #     subpsfim[:,-2:] = 0
    # 
    #     print('ph,pw, scale', ph,pw,scale)
    #     print('sh,sw', sh,sw)
    #     print('Subpsfim shape', subpsfim.shape)
    #     subpixpsf = PixelizedPSF(subpsfim[:-1,:-1])
    #     SP,(px0,py0),(spH,spW),(sw,sv) = subpixpsf.getFourierTransform(
    #         0., 0., scale*halfsize)
    # 
    #     wcs = NullWCS()
    #     wcs.pixscale /= scale
    #     print('subsampling image: set pixscale', wcs.pixscale)
    #     print('WCS:', wcs)
    # 
    #     subdata=np.zeros((scale*H,scale*W), np.float32)
    #     subimg = Image(data=subdata, invvar=np.ones_like(subdata), psf=tinypsf,
    #                    wcs=wcs)
    # 
    #     # Move galaxy to center of image.
    #     gal.pos.x = scale*pH/2.
    #     gal.pos.y = scale*pW/2.
    #     
    #     subtinyp = gal.getModelPatch(subimg)
    #     subtinypad = np.zeros((scale*pH,scale*pW))
    #     subtinyp.addTo(subtinypad)
    #     # Rotated to be zero-centered.
    #     subtinypad2 = np.fft.fftshift(subtinypad)
    #     
    #     Fsub = np.fft.rfft2(subtinypad2)
    #     tH,tW = subtinypad.shape
    #     Fsub /= (tH * np.pi)
    #     Fsub *= scale
    #     
    #     sz = scale * 32
    #     SG = np.fft.irfft2(Fsub * SP, s=(sz,sz))
    # 
    #     # Bin the subsampled ones...
    #     BSG = bin_image(SG, scale)
    #     BSG /= (scale**2)
    # 
    #     sz1 = 32
    #     MG = np.fft.irfft2(Fmine * P, s=(sz1,sz1))
    # 
    #     print('MG:')
    #     x1,y1 = measure(MG)
    #     #mx.append(x1)
    #     #my.append(y1)
    #     print('BSG:')
    #     x2,y2 = measure(BSG)
    #     #sx.append(x2)
    #     #sy.append(y2)
    #     dxlist.append(x2 - x1)
    #     dylist.append(y2 - y1)
    # 
    #     print('shift', x2-x1, y2-y1)
    #     shift = lanczos_shift_image(BSG, -(x2-x1), -(y2-y1), order=5)
    #     x3,y3 = measure(shift)
    #     print('shifted:', x3-x1, y3-y1)
    # 
    #     
    # plt.clf()
    # plt.plot(scales, dxlist, 'bo-')
    # plt.plot(scales, dylist, 'ro-')
    # ps3.savefig()
    # print('scales = np.array(', scales, ')')
    # print('dx = np.array(', dxlist, ')')
    # print('dy = np.array(', dylist, ')')
    # 
    # return
        
    
    scale = 8
    sh,sw = ph*scale, pw*scale
    subpsfim = np.zeros((sh,sw))
    step = 1./float(scale)
    for ix in np.arange(scale):
        for iy in np.arange(scale):
            dx = ix * step
            dy = iy * step
            #dx = 0.5 + (ix - 0.5) * step
            #dy = 0.5 + (iy - 0.5) * step
            #if ix == 0 and iy == 0:
            #    subpsfim[0::scale, 0::scale] = psfim
            #    continue
            shift = lanczos_shift_image(psfim, -dx, -dy, order=5)
            subpsfim[iy::scale, ix::scale] = shift

    # HACK -clamp edges to zero to counter possible weird lanczos edge issues?
    subpsfim[:2,:] = 0
    subpsfim[:,:2] = 0
    subpsfim[-2:,:] = 0
    subpsfim[:,-2:] = 0
            
    plt.clf()
    plt.subplot(1,2,1)
    dimshow(psfim)
    plt.subplot(1,2,2)
    dimshow(subpsfim)
    ps3.savefig()

    print('SubPSF image:', subpsfim.shape)
    subpixpsf = PixelizedPSF(subpsfim[:-1,:-1])
    SP,(px0,py0),(spH,spW),(sw,sv) = subpixpsf.getFourierTransform(
        0., 0., scale*halfsize)

    print('SP shape', SP.shape)
    
    wcs = NullWCS()
    wcs.pixscale /= scale
    print('subsampling image: set pixscale', wcs.pixscale)
    print('WCS:', wcs)
    
    subdata=np.zeros((scale*H,scale*W), np.float32)
    subimg = Image(data=subdata, invvar=np.ones_like(subdata), psf=tinypsf,
                   wcs=wcs)

    # Move galaxy to center of image.
    gal.pos.x = scale*pH/2.
    gal.pos.y = scale*pW/2.
    
    subtinyp = gal.getModelPatch(subimg)
    subtinypad = np.zeros((scale*pH,scale*pW))
    subtinyp.addTo(subtinypad)
    # Rotated to be zero-centered.
    subtinypad2 = np.fft.fftshift(subtinypad)
    
    plt.clf()
    dimshow(subtinypad2, **ima)
    plt.title('subtinypad2')
    ps3.savefig()

    plt.clf()
    dimshow(tinypad2, **ima)
    plt.title('tinypad2')
    ps3.savefig()
    
    Fsub = np.fft.rfft2(subtinypad2)
    tH,tW = subtinypad.shape
    Fsub /= (tH * np.pi)
    Fsub *= scale

    fima2 = fima.copy()
    fima2.update(vmin=0, vmax=1.1)

    Fsub_orig = Fsub.copy()

    # Trim Fsub to same size (63,33) vs (64,33)
    #h1,w1 = Fsub.shape
    #hm2,wm2 = Fmine2.shape
    #Fsub = Fsub[:hm2,:]
    
    plt.clf()
    dimshow(np.fft.fftshift(np.hypot(Fsub.real, Fsub.imag), axes=(0,)), **fima2)
    plt.colorbar()
    plt.title('Fsub')
    ps3.savefig()

    print('Fsub sum:', np.hypot(Fsub.real, Fsub.imag).sum())
    print('Fsub Real range:', Fsub.real.min(), Fsub.real.max())
    print('Fsub Imag range:', Fsub.imag.min(), Fsub.imag.max())

    pH,pW = subtinypad2.shape
    wt = np.fft.rfftfreq(pW)
    vt = np.fft.fftfreq(pH)
    # print('wsub:', len(wt), 'min/max', wt.min(), wt.max())
    # print('vsub:', len(vt), 'min/max', vt.min(), vt.max())

    df = np.abs(w[1] - w[0])
    w2b = np.arange(scale * (len(w)-1) + 1) * df + scale*np.min(w)
    v2b = np.arange(scale*len(v)) * df + scale*np.min(v)
    # print('w2:', w2)
    # print('w2b:', w2b)
    # print('v2:', v2)
    # print('v2b:', v2b)
    Fmine2b = amix.getFourierTransform(w2b, v2b)
    # -> shape len(v2b),len(w2b)
    
    print('Fmine2b shape', Fmine2b.shape)
    print('Fmine2b sum:', np.hypot(Fmine2b.real, Fmine2b.imag).sum())
    print('Fmine2b Real range:', Fmine2b.real.min(), Fmine2b.real.max())
    print('Fmine2b Imag range:', Fmine2b.imag.min(), Fmine2b.imag.max())

    print('Fsub_orig shape:', Fsub_orig.shape)
    
    # # My method, Fourier transform with twice the frequency range
    # plt.clf()
    # dimshow(np.hypot(Fmine2.real, Fmine2.imag), **fima2)
    # plt.colorbar()
    # plt.title('Fmine2')
    # ps3.savefig()
    # 
    # diff = np.fft.fftshift(Fsub, axes=(0,)) - Fmine2
    # print('diff Real range:', diff.real.min(), diff.real.max())
    # print('diff Imag range:', diff.imag.min(), diff.imag.max())
    # 
    # plt.clf()
    # dimshow(np.hypot(diff.real, diff.imag), **fima)
    # plt.colorbar()
    # plt.title('Fsub - Fmine2')
    # ps3.savefig()

    diff2 = np.fft.fftshift(Fsub_orig, axes=(0,)) - Fmine2b
    print('diff2 Real range:', diff.real.min(), diff.real.max())
    print('diff2 Imag range:', diff.imag.min(), diff.imag.max())
    
    plt.clf()
    dimshow(np.hypot(diff2.real, diff2.imag), **fima)
    plt.colorbar()
    plt.title('Fsub - Fmine2b')
    ps3.savefig()

    plt.clf()
    dimshow(diff2.real, **rima)
    plt.colorbar()
    plt.title('real(Fsub - Fmine2b)')
    ps3.savefig()

    print('Ftiny:', Ftiny.shape)
    print('P:', P.shape)
    print('Fsub:', Fsub.shape)
    print('SP:', SP.shape)
    print('Fmine2b:', Fmine2b.shape)
    
    #sz1 = 32
    #sz = 64
    sz1 = 32
    sz = scale * 32
    
    plt.clf()
    plt.subplot(2,2,1)
    dimshow(Ftiny.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,2)
    dimshow(Ftiny.imag, **iima)
    plt.colorbar()
    plt.subplot(2,2,3)
    dimshow(P.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,4)
    dimshow(P.imag, **iima)
    plt.colorbar()
    plt.suptitle('Ftiny, P')
    ps3.savefig()
    
    PG = np.fft.irfft2(Ftiny * P, s=(sz1,sz1))
    print('PG', PG.dtype, PG.shape)
    plt.clf()
    dimshow(PG, **ima)
    plt.colorbar()
    plt.title('PG')
    ps3.savefig()
    
    #P,(px0,py0),(pH,pW),(w,v) = pixpsf.getFourierTransform(0., 0., halfsize)
    #Fmine = amix.getFourierTransform(w, v)

    plt.clf()
    plt.subplot(2,2,1)
    dimshow(Fmine.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,2)
    dimshow(Fmine.imag, **iima)
    plt.colorbar()
    plt.subplot(2,2,3)
    dimshow(P.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,4)
    dimshow(P.imag, **iima)
    plt.colorbar()
    plt.suptitle('Fmine, P')
    ps3.savefig()

    MG = np.fft.irfft2(Fmine * P, s=(sz1,sz1))
    print('MG', MG.dtype, MG.shape)
    plt.clf()
    dimshow(MG, **ima)
    plt.colorbar()
    plt.title('MG')
    ps3.savefig()

    plt.clf()
    diffshow(PG - MG, **ima)
    plt.colorbar()
    plt.title('PG - MG')
    ps3.savefig()
    
    plt.clf()
    plt.subplot(2,2,1)
    dimshow(Fsub.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,2)
    dimshow(Fsub.imag, **iima)
    plt.colorbar()
    plt.subplot(2,2,3)
    dimshow(SP.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,4)
    dimshow(SP.imag, **iima)
    plt.colorbar()
    plt.suptitle('Fsub, SP')
    ps3.savefig()

    SG = np.fft.irfft2(Fsub * SP, s=(sz,sz))
    print('SG', SG.dtype, SG.shape)
    plt.clf()
    dimshow(SG, **ima)
    plt.colorbar()
    plt.title('SG')
    ps3.savefig()

    Fmine2c = np.fft.fftshift(Fmine2b, axes=(0,))
    
    plt.clf()
    plt.subplot(2,2,1)
    dimshow(Fmine2c.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,2)
    dimshow(Fmine2c.imag, **iima)
    plt.colorbar()
    plt.subplot(2,2,3)
    dimshow(SP.real, **rima)
    plt.colorbar()
    plt.subplot(2,2,4)
    dimshow(SP.imag, **iima)
    plt.colorbar()
    plt.suptitle('Fmine2c, SP')
    ps3.savefig()

    MG2 = np.fft.irfft2(Fmine2c * SP, s=(sz,sz))
    print('MG2', MG2.dtype, MG2.shape)
    plt.clf()
    dimshow(MG2, **ima)
    plt.colorbar()
    plt.title('MG2')
    ps3.savefig()

    plt.clf()
    diffshow(SG - MG2, **ima)
    plt.colorbar()
    plt.title('SG - MG2')
    ps3.savefig()

    # Bin the subsampled ones...
    BSG = bin_image(SG, scale)
    BSG /= (scale**2)

    print('MG:')
    x1,y1 = measure(MG)
    print('BSG:')
    x2,y2 = measure(BSG)

    shiftBSG = lanczos_shift_image(BSG, -(x2-x1), -(y2-y1), order=5)
    
    plt.clf()
    dimshow(BSG, **ima)
    plt.colorbar()
    plt.title('BSG')
    ps3.savefig()

    plt.clf()
    diffshow(BSG - MG, **ima)
    plt.colorbar()
    plt.title('BSG - MG')
    ps3.savefig()

    plt.clf()
    diffshow(shiftBSG - MG, **ima)
    plt.colorbar()
    plt.title('shiftBSG - MG')
    ps3.savefig()

    

def measure(img):
    from scipy.ndimage.measurements import center_of_mass
    #cy,cx = center_of_mass(img)
    #print('Center of mass: %.2f, %.2f' % (cx,cy))
    h,w = img.shape
    xx,yy = np.meshgrid(np.arange(w), np.arange(h))
    isum = img.sum()
    cx,cy = np.sum(xx * img) / isum, np.sum(yy * img) / isum
    #cr = np.sqrt(np.sum(((xx - cx)**2 + (yy - cy)**2) * img) / isum)
    print('Center of mass: %.2f, %.2f' % (cx,cy))
    #print('Second moment: %.2f' % cr)
    return cx,cy
    
def bin_image(data, S):
    # rebin image data
    H,W = data.shape
    sH,sW = (H+S-1)/S, (W+S-1)/S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    for i in range(S):
        for j in range(S):
            subdata = data[i::S, j::S]
            subh,subw = subdata.shape
            newdata[:subh,:subw] += subdata
    return newdata
    
    
def main():
    # !!important!!
    disable_galaxy_cache()

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
    #plt.figure(2, figsize=(frac * 3,3))
    #plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

    plt.figure(2, figsize=(6,6))
    
    # plt.subplots_adjust(left=margin/frac, right=1 - margin/frac,
    #                     bottom=margin, top=1.-margin,
    #                     hspace=0, wspace=0)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

    
    #plt.figure(3, figsize=(3,6))
    plt.figure(3, figsize=(6 * frac,6))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    lopass(psfex, ps3)
    sys.exit(0)
    
    plt.figure(1)
    fft_plots(psfex, W, H, ps, ps2)

    
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
if __name__ == '__main__':
    main()
