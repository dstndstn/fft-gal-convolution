from __future__ import print_function
import numpy as np

from demo import galaxy_psf_convolution, GaussianGalaxy
import galsim

def measure(img):
    from scipy.ndimage.measurements import center_of_mass
    #cy,cx = center_of_mass(img)
    #print('Center of mass: %.2f, %.2f' % (cx,cy))
    h,w = img.shape
    xx,yy = np.meshgrid(np.arange(w), np.arange(h))
    isum = img.sum()
    cx,cy = np.sum(xx * img) / isum, np.sum(yy * img) / isum
    cr = np.sqrt(np.sum(((xx - cx)**2 + (yy - cy)**2) * img) / isum)
    print('Center of mass: %.2f, %.2f' % (cx,cy))
    print('Second moment: %.2f' % cr)

class plotset(object):
    def __init__(self, prefix):
        self.n = 0
        self.prefix = prefix
    def savefig(self):
        plt.savefig(self.prefix + '%02i.png' % self.n)
        self.n += 1
    
def compare(ph, pw, psf_sigma, gal_sigma, ps):
    pixscale = 1.
    cd = pixscale * np.eye(2) / 3600.

    cx,cy = pw/2, ph/2
    xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))

    # Create pixelized PSF (Gaussian)
    pixpsf = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / psf_sigma**2)

    plt.clf()
    plt.imshow(pixpsf, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('PSF')
    plt.colorbar()
    ps.savefig()

    G = galaxy_psf_convolution(gal_sigma, 0., 0., GaussianGalaxy, cd,
                               0., 0., pixpsf)
    G /= G.sum()
    
    plt.clf()
    plt.imshow(G, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Convolved')
    plt.colorbar()
    ps.savefig()

    print('Convolved:')
    measure(G)
    
    gal = galsim.Gaussian(flux=1., sigma=gal_sigma)
    psf = galsim.Gaussian(flux=1., sigma=psf_sigma)
    final = galsim.Convolve([gal, psf])
    print('Final:', final)
    image = galsim.ImageF(pw,ph)
    final.drawImage(image, offset=(0.5, 0.5), scale=pixscale, method='sb')
    gs = image.array
    gs /= gs.sum()
    print('Galsim shifted:')
    measure(gs)

    
    plt.clf()
    plt.imshow(gs, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('GalSim analytic')
    plt.colorbar()
    ps.savefig()
    
    # Create pixelized PSF (x) Gaussian
    gal_psf_sigma = np.hypot(psf_sigma, gal_sigma)
    Gpix = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / gal_psf_sigma**2)
    Gpix /= Gpix.sum()

    plt.clf()
    plt.imshow(Gpix, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Analytic')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    plt.imshow(G - Gpix, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Convolved - Analytic')
    plt.colorbar()
    ps.savefig()

    # Create pixelized galaxy
    gx,gy = cx,cy
    pixgal = np.exp(-0.5 * ((xx-gx)**2 + (yy-gy)**2) / gal_sigma**2)

    # FFT convolution
    Fpsf = np.fft.rfft2(pixpsf)
    print('Fpsf:', Fpsf.shape, Fpsf.dtype)
    Fgal = np.fft.rfft2(pixgal)
    print('Fgal:', Fgal.shape, Fgal.dtype)
    Gconv = np.fft.irfft2(Fpsf * Fgal)
    Gconv = np.fft.ifftshift(Gconv)
    print('Gconv:', Gconv.shape, Gconv.dtype)
    Gconv /= Gconv.sum()

    print('Pixelized:')
    measure(Gconv)

    plt.clf()
    plt.imshow(Gconv, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Pixelized convolution')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    plt.imshow(G - Gconv, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Convolved - Pixelized')
    plt.colorbar()
    ps.savefig()


    
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    ps = plotset('c')
    
    # PSF (and convolved galaxy) image size
    ph,pw = 64,64

    psf_sigma = 1.

    compare(ph, pw, psf_sigma, gal_sigma, ps)

    
