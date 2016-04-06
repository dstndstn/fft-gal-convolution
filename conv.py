from __future__ import print_function
import scipy.stats
from pylab import *
import numpy as np

from astrometry.util.plotutils import PlotSequence, dimshow

from compare import measure
    

def integrate_gaussian(G, xx, yy):
    Gcdf = G.cdf(xx) * G.cdf(yy)
    Gcdf = Gcdf[1:,1:] + Gcdf[:-1,:-1] - Gcdf[1:,:-1] - Gcdf[:-1,1:]
    return Gcdf

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

ps = PlotSequence('conv')

S = 51
center = S/2
print('Center', center)
# psf
psf = scipy.stats.norm(loc=center + 0.5, scale=2.)

gal_sigma = 3.

x = np.arange(S)
xx,yy = np.meshgrid(x, np.arange(S))

Pcdf = psf.cdf(xx) * psf.cdf(yy)

# plt.clf()
# plt.imshow(Pcdf, interpolation='nearest', origin='lower')
# ps.savefig()

pixpsf = integrate_gaussian(psf, xx, yy)

# plt.clf()
# plt.imshow(Gcdf, interpolation='nearest', origin='lower')
# plt.savefig('dcdf.png')

# plt.clf()
# plt.imshow(np.exp(-0.5 * ((xx-center)**2 + (yy-center)**2)/2.**2),
#            interpolation='nearest', origin='lower')
# plt.savefig('g.png')

# my convolution
from demo import galaxy_psf_convolution, GaussianGalaxy
pixscale = 1.
cd = pixscale * np.eye(2) / 3600.
P,FG,Gmine = galaxy_psf_convolution(gal_sigma, 0., 0., GaussianGalaxy, cd,
                                    0., 0., pixpsf, debug=True)

subsample = [1,2,3,4]

for s in subsample:

    print()
    print('Subsample', s)
    print()

    step = 1./s

    sz = s * (S-1) + 1
    
    xx,yy = np.meshgrid(np.arange(0, S, step)[:sz+1],
                        np.arange(0, S, step)[:sz+1])
    # Create pixelized PSF (Gaussian)
    subpixpsf = integrate_gaussian(psf, xx, yy)

    binned = bin_image(subpixpsf, s)

    bh,bw = binned.shape
    pixpsf1 = pixpsf[:bh,:bw]
    ph,pw = pixpsf.shape
    binned = binned[:ph,:pw]
    
    plt.clf()

    plt.subplot(2,2,1)
    dimshow(subpixpsf)
    plt.title('subpix psf')
    plt.colorbar()

    plt.subplot(2,2,2)
    dimshow(binned)
    plt.title('binned subpix psf')
    plt.colorbar()

    plt.subplot(2,2,3)
    dimshow(pixpsf1)
    plt.title('pix psf')
    plt.colorbar()

    plt.subplot(2,2,4)
    dimshow(pixpsf1 - binned)
    plt.title('pix - binned')
    plt.colorbar()
    plt.suptitle('subsample %i' % s)
    ps.savefig()
    
    # Create pixelized galaxy image
    #gxx,gyy = xx + step/2., yy + step/2.
    gxx,gyy = xx,yy
    subpixgal = np.exp(-0.5 * ((gxx-center)**2 + (gyy-center)**2)/gal_sigma**2)
    sh,sw = subpixpsf.shape
    subpixgal = subpixgal[:sh,:sw]

    print('Subpix psf, gal', subpixpsf.shape, subpixgal.shape)

    print('Subpix PSF:')
    measure(subpixpsf)
    print('Subpix gal:')
    measure(subpixgal)
    
    # FFT convolution
    Fpsf = np.fft.rfft2(subpixpsf)
    spg = np.fft.ifftshift(subpixgal)

    # plt.clf()
    # dimshow(spg)
    # plt.title('spg')
    # ps.savefig()


    Fgal = np.fft.rfft2(spg)
    Fconv = Fpsf * Fgal
    subpixfft = np.fft.irfft2(Fconv, s=subpixpsf.shape)
    print('Shapes:', 'subpixpsf', subpixpsf.shape, 'Fpsf', Fpsf.shape)
    print('spg', spg.shape, 'Fgal', Fgal.shape, 'Fconv', Fconv.shape,
          'subpixfft', subpixfft.shape)
    
    print('Subpix conv', subpixfft.shape)
    
    binned = bin_image(subpixfft, s)
    binned /= np.sum(binned)

    print('Binned', binned.shape)
    print('Mine:', Gmine.shape)
    print('Mine:')
    measure(Gmine)
    print('Binned subpix FFT:')
    measure(binned)

    mh,mw = Gmine.shape
    binned = binned[:mh,:mw]
    
    plt.clf()

    plt.subplot(2,3,1)
    dimshow(subpixpsf)
    plt.title('subpix psf')
    plt.colorbar()

    plt.subplot(2,3,2)
    dimshow(subpixgal)
    plt.title('subpix galaxy')
    plt.colorbar()

    plt.subplot(2,3,3)
    dimshow(subpixfft)
    plt.title('subpix FFT conv')
    plt.colorbar()

    plt.subplot(2,3,4)
    dimshow(binned)
    plt.title('binned FFT conv')
    plt.colorbar()

    plt.subplot(2,3,5)
    dimshow(Gmine)
    plt.title('my conv')
    plt.colorbar()
    
    plt.subplot(2,3,6)
    dimshow(Gmine - binned)
    plt.title('my conv')
    plt.colorbar()
    
    ps.savefig()
