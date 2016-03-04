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

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    # PSF (and convolved galaxy) image size
    ph,pw = 64,64

    xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
    cx,cy = pw/2, ph/2

    psf_sigma = 1.
    
    # Create pixelized PSF (Gaussian)
    pixpsf = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / psf_sigma**2)

    plt.clf()
    plt.imshow(pixpsf, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('PSF')
    plt.colorbar()
    plt.savefig('c0.png')

    re = 1.
    cd = np.eye(2) / 3600.
    
    G = galaxy_psf_convolution(re, 0., 0., GaussianGalaxy, cd, 0., 0., pixpsf)
    G /= G.sum()
    
    plt.clf()
    plt.imshow(G, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Convolved')
    plt.colorbar()
    plt.savefig('c1.png')

    print('Convolved:')
    measure(G)
    
    gal = galsim.Gaussian(flux=1., sigma=re)
    psf = galsim.Gaussian(flux=1., sigma=psf_sigma)
    final = gal
    #final = galsim.Convolve([gal, psf])
    #image = final.draw(dx=1.)
    #final = final.shift(dx=0.5, dy=0.5)
    print('Final:', final)
    image = galsim.ImageF(pw,ph)
    final.drawImage(image) #, method='sb')
    gs = image.array
    gs /= gs.sum()

    print('Galsim:')
    measure(gs)

    # ??? Shifting by 0.5,0.5 does not shift the centroid by that amount!!
    
    final = final.shift(dx=0.5, dy=0.5)
    print('Final:', final)
    image = galsim.ImageF(pw,ph)
    final.drawImage(image) #, method='sb')
    gs = image.array
    gs /= gs.sum()
    print('Galsim shifted:')
    measure(gs)

    
    plt.clf()
    plt.imshow(gs, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('GalSim analytic')
    plt.colorbar()
    plt.savefig('c2.png')
    
    # Create pixelized PSF (x) Gaussian
    gal_psf_sigma = np.hypot(psf_sigma, re)

    Gpix = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / gal_psf_sigma**2)
    Gpix /= Gpix.sum()

    plt.clf()
    plt.imshow(Gpix, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Analytic')
    plt.colorbar()
    plt.savefig('c3.png')
    
    plt.clf()
    plt.imshow(G - Gpix, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Convolved - Analytic')
    plt.colorbar()
    plt.savefig('c4.png')

    
