import numpy as np

from demo import galaxy_psf_convolution, GaussianGalaxy

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
    
    # Create pixelized PSF (x) Gaussian
    gal_psf_sigma = np.hypot(psf_sigma, re)

    Gpix = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / gal_psf_sigma**2)
    Gpix /= Gpix.sum()

    plt.clf()
    plt.imshow(Gpix, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Analytic')
    plt.colorbar()
    plt.savefig('c2.png')
    
    plt.clf()
    plt.imshow(G - Gpix, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Convolved - Analytic')
    plt.colorbar()
    plt.savefig('c3.png')

    
