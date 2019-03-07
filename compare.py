from __future__ import print_function
import numpy as np

from demo import galaxy_psf_convolution, GaussianGalaxy

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

def imshow_diff(d, title):
    mx = np.max(np.abs(d))
    plt.imshow(d, interpolation='nearest', origin='lower', cmap='gray',
               vmin=-mx, vmax=mx)
    plt.title(title + ': L2err %.2g' % (np.sqrt(np.sum(d**2))))
        
def compare(ph, pw, psf_sigma, gal_sigma, ps, subsample):
    import galsim

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

    #Gmine = galaxy_psf_convolution(gal_sigma, 0., 0., GaussianGalaxy, cd,
    #                           0., 0., pixpsf)

    P,FG,Gmine,v,w = galaxy_psf_convolution(gal_sigma, 0., 0., GaussianGalaxy,
                                            cd, 0., 0., pixpsf, debug=True)

    Gmine /= Gmine.sum()
    
    plt.clf()
    plt.imshow(Gmine, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Mine')
    plt.colorbar()
    ps.savefig()

    print('Mine:')
    measure(Gmine)

    # Create pixelized PSF (x) Gaussian
    gal_psf_sigma = np.hypot(psf_sigma, gal_sigma)
    Gana = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / gal_psf_sigma**2)
    Gana /= Gana.sum()
    print('Pixelized, analytic:')
    measure(Gana)
    
    gal = galsim.Gaussian(flux=1., sigma=gal_sigma)
    psf = galsim.Gaussian(flux=1., sigma=psf_sigma)
    final = galsim.Convolve([gal, psf])
    print('Final:', final)
    image = galsim.ImageF(pw,ph)
    final.drawImage(image, offset=(0.5, 0.5), scale=pixscale, method='sb')
    Ggs = image.array
    Ggs /= Ggs.sum()
    print('Galsim shifted:')
    measure(Ggs)

    
    plt.clf()
    plt.imshow(Ggs, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('GalSim analytic')
    plt.colorbar()
    ps.savefig()

    plt.clf()
    imshow_diff(Ggs - Gana, 'GalSim Analytic - Analytic')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    plt.imshow(Gana, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Analytic')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    imshow_diff(Gmine - Gana, 'Mine - Analytic')
    plt.colorbar()
    ps.savefig()

    r = np.hypot(xx - cx, yy - cy)
    plt.clf()
    plt.subplot(2,1,1)
    plt.loglog(r.ravel(), Gmine.ravel(), 'b.')
    plt.loglog(r.ravel(), Gana.ravel(), 'g.')
    plt.ylim(1e-20, 1.)
    plt.ylabel('mine(b) analytic(g)')
    plt.subplot(2,1,2)
    plt.plot(r.ravel(), (Gmine-Gana).ravel(), 'r.')
    plt.ylabel('mine - analytic')
    plt.xlabel('radius (pix)')
    ps.savefig()

    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(np.log10(np.maximum(Gana, 1e-16)),
               interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Log Analytic')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(np.log10(np.maximum(Gmine, 1e-16)),
               interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Log Mine')
    plt.colorbar()
    ps.savefig()
    
    # Create pixelized galaxy
    gx,gy = cx,cy
    pixgal = np.exp(-0.5 * ((xx-gx)**2 + (yy-gy)**2) / gal_sigma**2)
    pixgal /= np.sum(pixgal)
    
    # FFT convolution
    Fpsf = np.fft.rfft2(pixpsf)
    print('Fpsf:', Fpsf.shape, Fpsf.dtype)

    spg = np.fft.ifftshift(pixgal)
    Fgal = np.fft.rfft2(spg)
    #Fgal = np.fft.rfft2(pixgal)

    print('Fgal:', Fgal.shape, Fgal.dtype)

    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(Fpsf.real, interpolation='nearest', origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('Fpsf.real')
    plt.subplot(1,2,2)
    plt.imshow(Fpsf.imag, interpolation='nearest', origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('Fpsf.imag')
    ps.savefig()

    print('P:', P.shape, P.dtype)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(P.real, interpolation='nearest', origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('P.real')
    plt.subplot(1,2,2)
    plt.imshow(P.imag, interpolation='nearest', origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('P.imag')
    ps.savefig()


    
    psf2 = np.fft.irfft2(Fpsf)

    plt.clf()
    plt.imshow(psf2 - pixpsf, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('ifft(fft(psf)) - psf')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(Fgal.real, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Fgal.real [%.2g,%.2g]' % (Fgal.real.min(), Fgal.real.max()))
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(Fgal.imag, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Fgal.imag [%.2g,%.2g]' % (Fgal.imag.min(), Fgal.imag.max()))
    plt.colorbar()
    ps.savefig()

    print('FG:', FG.shape, FG.dtype)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(FG.real, interpolation='nearest', origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('FG.real [%.2g,%.2g]' % (FG.real.min(), FG.real.max()))
    plt.subplot(1,2,2)
    plt.imshow(FG.imag, interpolation='nearest', origin='lower', cmap='gray')
    plt.colorbar()
    plt.title('FG.imag [%.2g,%.2g]' % (FG.imag.min(), FG.imag.max()))
    ps.savefig()

    
    Fconv = Fpsf * Fgal
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(Fconv.real, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Fconv.real')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(Fconv.imag, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Fconv.imag')
    plt.colorbar()
    ps.savefig()

    Fconvmine = P * FG
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(Fconvmine.real, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Fconvmine.real')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(Fconvmine.imag, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Fconvmine.imag')
    plt.colorbar()
    ps.savefig()
    
    Fgal.imag[:,:] = 0.
    
    Gfft = np.fft.irfft2(Fpsf * Fgal)
    #Gfft = np.fft.ifftshift(Gfft)
    print('Gfft:', Gfft.shape, Gfft.dtype)
    Gfft /= Gfft.sum()
    
    print('Pixelized (fft):')
    measure(Gfft)

    plt.clf()
    plt.imshow(Gfft, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Pixelized convolution')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    imshow_diff(Gfft - Gana, 'Pixelized - Analytic')
    plt.colorbar()
    ps.savefig()

    plt.clf()
    imshow_diff(Gmine - Gfft, 'Mine - Pixelized')
    plt.colorbar()
    ps.savefig()
    
    for s in subsample:

        step = 1./s

        #xx,yy = np.meshgrid(-step/2. + np.arange(0, pw, step),
        #                    -step/2. + np.arange(0, ph, step))

        xx,yy = np.meshgrid(-0.5 + step/2. + np.arange(0, pw, step),
                            -0.5 + step/2. + np.arange(0, ph, step))
        
        # Create pixelized PSF (Gaussian)
        #pixpsf = np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / psf_sigma**2)

        subpixpsf = np.repeat(np.repeat(pixpsf, s, axis=0), s, axis=1)
        
        gxx,gyy = np.meshgrid(np.arange(0, pw*s),
                              np.arange(0, ph*s))
        gx,gy = pw*s/2, ph*s/2
        pixgal = np.exp(-0.5 * ((gxx-gx)**2 + (gyy-gy)**2) / (gal_sigma*s)**2)

        #gx,gy = cx - step/2., cy - step/2.
        #pixgal = np.exp(-0.5 * ((xx-gx)**2 + (yy-gy)**2) / gal_sigma**2)
        # FFT convolution
        Fpsf = np.fft.rfft2(subpixpsf)
        print('Fpsf:', Fpsf.shape, Fpsf.dtype)
        Fgal = np.fft.rfft2(pixgal)
        print('Fgal:', Fgal.shape, Fgal.dtype)
        Gfft = np.fft.irfft2(Fpsf * Fgal)
        Gfft = np.fft.ifftshift(Gfft)
        print('Gfft:', Gfft.shape, Gfft.dtype)
        Gfft /= Gfft.sum()
        print('Subsampled, pixelized:')
        measure(Gfft)
    
        plt.clf()
        plt.imshow(Gfft, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('Subsampled convolution')
        plt.colorbar()
        ps.savefig()
        
        # Bin down s x s
        Gbin = np.zeros((ph, pw))
        for i in range(s):
            for j in range(s):
                Gbin += Gfft[i::s, j::s]
        Gfft = Gbin
        Gfft /= Gfft.sum()
    
        print('Subsampled, pixelized:')
        measure(Gfft)
        
        plt.clf()
        plt.imshow(Gfft, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('Subsampled & binned convolution')
        plt.colorbar()
        ps.savefig()
        
        plt.clf()
        imshow_diff(Gfft - Gana, 'Sub-Pixelized - Analytic')
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
    gal_sigma = 1.
    compare(ph, pw, psf_sigma, gal_sigma, ps, [])

    # PSF (and convolved galaxy) image size
    # ph,pw = 128,128
    # psf_sigma = 1.
    # gal_sigma = 1.
    # compare(ph, pw, psf_sigma, gal_sigma, ps, [])
    
    psf_sigma = 1.
    # undersampled
    gal_sigma = 0.5
    compare(ph, pw, psf_sigma, gal_sigma, ps, [2,4])
    
