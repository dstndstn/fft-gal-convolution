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


def compare_subsampled(S, s, ps, psf, pixpsf, Gmine, gal_sigma, psf_sigma,
                       get_ffts=False):
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
    
    # plt.clf()
    # 
    # plt.subplot(2,2,1)
    # dimshow(subpixpsf)
    # plt.title('subpix psf')
    # plt.colorbar()
    # 
    # plt.subplot(2,2,2)
    # dimshow(binned)
    # plt.title('binned subpix psf')
    # plt.colorbar()
    # 
    # plt.subplot(2,2,3)
    # dimshow(pixpsf1)
    # plt.title('pix psf')
    # plt.colorbar()
    # 
    # plt.subplot(2,2,4)
    # dimshow(pixpsf1 - binned)
    # plt.title('pix - binned')
    # plt.colorbar()
    # plt.suptitle('subsample %i' % s)
    # ps.savefig()
    
    # Create pixelized galaxy image
    #gxx,gyy = xx + step/2., yy + step/2.
    gxx,gyy = xx,yy
    center = S/2
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

    if get_ffts:
        return Fpsf, Fgal

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
    dimshow(np.log10(np.maximum(binned / np.max(binned), 1e-12)))
    plt.title('binned FFT conv')
    plt.colorbar()

    plt.subplot(2,3,5)
    dimshow(np.log10(np.maximum(Gmine / np.max(Gmine), 1e-12)))
    #dimshow(Gmine)
    plt.title('my conv')
    plt.colorbar()

    gh,gw = Gmine.shape
    binned = binned[:gh,:gw]
    bh,bw = binned.shape
    Gmine = Gmine[:bh,:bw]
    diff  = Gmine - binned
    
    plt.subplot(2,3,6)
    dimshow(diff)
    plt.title('mine - FFT')
    plt.colorbar()

    plt.suptitle('PSF %g, Gal %g, subsample %i' % (psf_sigma, gal_sigma, s))
    
    ps.savefig()

    rmsdiff = np.sqrt(np.mean(diff**2))
    return rmsdiff


def main():
    ps = PlotSequence('conv')
    
    S = 51
    center = S/2
    print('Center', center)

    for psf_sigma in [2., 1.5, 1.]:

        # psf
        psf = scipy.stats.norm(loc=center + 0.5, scale=psf_sigma)

        rms2 = []

        x = np.arange(S)
        xx,yy = np.meshgrid(x, np.arange(S))
        Pcdf = psf.cdf(xx) * psf.cdf(yy)
        # plt.clf()
        # plt.imshow(Pcdf, interpolation='nearest', origin='lower')
        # ps.savefig()
        pixpsf = integrate_gaussian(psf, xx, yy)

        padpsf = np.zeros((S*2-1, S*2-1))
        ph,pw = pixpsf.shape
        padpsf[S/2:S/2+ph, S/2:S/2+pw] = pixpsf
        Fpsf = np.fft.rfft2(padpsf)

        padh,padw = padpsf.shape
        v = np.fft.rfftfreq(padw)
        w = np.fft.fftfreq(padh)
        fmax = max(max(np.abs(v)), max(np.abs(w)))
        cut = fmax / 2. * 1.000001
        #print('Frequence cut:', cut)
        Ffiltpsf = Fpsf.copy()
        #print('Ffiltpsf', Ffiltpsf.shape)
        #print((np.abs(w) < cut).shape)
        #print((np.abs(v) < cut).shape)
        Ffiltpsf[np.abs(w) > cut, :] = 0.
        Ffiltpsf[:, np.abs(v) > cut] = 0.
        #print('pad v', v)
        #print('pad w', w)

        filtpsf = np.fft.irfft2(Ffiltpsf, s=(padh,padw))

        print('filtered PSF real', np.max(np.abs(filtpsf.real)))
        print('filtered PSF imag', np.max(np.abs(filtpsf.imag)))
        
        plt.clf()
        plt.subplot(2,3,1)
        dimshow(Fpsf.real)
        plt.colorbar()
        plt.title('Padded PSF real')
        plt.subplot(2,3,4)
        dimshow(Fpsf.imag)
        plt.colorbar()
        plt.title('Padded PSF imag')

        plt.subplot(2,3,2)
        dimshow(Ffiltpsf.real)
        plt.colorbar()
        plt.title('Filt PSF real')
        plt.subplot(2,3,5)
        dimshow(Ffiltpsf.imag)
        plt.colorbar()
        plt.title('Filt PSF imag')

        plt.subplot(2,3,3)
        dimshow(filtpsf.real)
        plt.title('PSF real')
        plt.colorbar()

        plt.subplot(2,3,6)
        dimshow(filtpsf.imag)
        plt.title('PSF imag')
        plt.colorbar()
        
        ps.savefig()


        pixpsf = filtpsf
        
        
        gal_sigmas = [2, 1, 0.5, 0.25]
        for gal_sigma in gal_sigmas:
    
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
            P,FG,Gmine,v,w = galaxy_psf_convolution(
                gal_sigma, 0., 0., GaussianGalaxy, cd,
                0., 0., pixpsf, debug=True)

            #print('v:', v)
            #print('w:', w)
            #print('P:', P.shape)

            print()
            print('PSF %g, Gal %g' % (psf_sigma, gal_sigma))
            
            rmax = np.argmax(np.abs(w))
            cmax = np.argmax(np.abs(v))
            l2_rmax = np.sqrt(np.sum(P[rmax,:].real**2 + P[rmax,:].imag**2))
            l2_cmax = np.sqrt(np.sum(P[:,cmax].real**2 + P[:,cmax].imag**2))
            print('PSF L_2 in highest-frequency rows & cols:', l2_rmax, l2_cmax)

            l2_rmax = np.sqrt(np.sum(FG[rmax,:].real**2 + FG[rmax,:].imag**2))
            l2_cmax = np.sqrt(np.sum(FG[:,cmax].real**2 + FG[:,cmax].imag**2))
            print('Gal L_2 in highest-frequency rows & cols:', l2_rmax, l2_cmax)

            C = P * FG
            l2_rmax = np.sqrt(np.sum(C[rmax,:].real**2 + C[rmax,:].imag**2))
            l2_cmax = np.sqrt(np.sum(C[:,cmax].real**2 + C[:,cmax].imag**2))
            print('PSF*Gal L_2 in highest-frequency rows & cols:', l2_rmax, l2_cmax)
            print()

            plt.clf()
            plt.subplot(2,4,1)
            dimshow(P.real)
            plt.colorbar()
            plt.title('PSF real')
            plt.subplot(2,4,5)
            dimshow(P.imag)
            plt.colorbar()
            plt.title('PSF imag')

            plt.subplot(2,4,2)
            dimshow(FG.real)
            plt.colorbar()
            plt.title('Gal real')
            plt.subplot(2,4,6)
            dimshow(FG.imag)
            plt.colorbar()
            plt.title('Gal imag')

            plt.subplot(2,4,3)
            dimshow((P * FG).real)
            plt.colorbar()
            plt.title('P*Gal real')
            plt.subplot(2,4,7)
            dimshow((P * FG).imag)
            plt.colorbar()
            plt.title('P*Gal imag')

            Fpsf, Fgal = compare_subsampled(
                S, 1, ps, psf, pixpsf, Gmine,
                gal_sigma, psf_sigma, get_ffts=True)
            
            plt.subplot(2,4,4)
            dimshow((Fgal).real)
            plt.colorbar()
            plt.title('pixGal real')
            plt.subplot(2,4,8)
            dimshow((Fgal).imag)
            plt.colorbar()
            plt.title('pixGal imag')
            
            plt.suptitle('PSF %g, Gal %g' % (psf_sigma, gal_sigma))

            ps.savefig()
            continue
        
            
            subsample = [1,2,3,4,5]
            rms1 = []
            for s in subsample:
                rms = compare_subsampled(S, s, ps, psf, pixpsf, Gmine, gal_sigma, psf_sigma)
                rms1.append(rms)
            rms2.append(rms1)


        print()
        print('PSF sigma =', psf_sigma)
        print('RMSes:')
        for rms1,gal_sigma in zip(rms2, gal_sigmas):
            print('Gal sigma', gal_sigma, 'rms:',
                  ', '.join(['%.3g' % r for r in rms1]))
                

if __name__ == '__main__':
    main()


'''

PSF sigma = 2.0
RMSes:
Gal sigma 2 rms: [2.86e-18, 2.37e-17, 8.18e-17, 1.03e-16, 8.54e-17]
Gal sigma 1 rms: [6.56e-10, 4.06e-15, 4.06e-15, 4.06e-15, 4.06e-15]
Gal sigma 0.5 rms: [1.73e-05, 3.8e-11, 4.17e-14, 4.17e-14, 4.17e-14]
Gal sigma 0.25 rms: [2.96e-05, 4.25e-06, 2.11e-08, 7e-12, 2.66e-14]

PSF sigma = 1.5
RMSes:
Gal sigma 2 rms: 7.71e-17, 9.23e-16, 9.28e-16, 9.23e-16, 9.34e-16
Gal sigma 1 rms: 6.15e-09, 6.59e-11, 6.59e-11, 6.59e-11, 6.59e-11
Gal sigma 0.5 rms: 4.07e-05, 6.8e-10, 6.72e-10, 6.72e-10, 6.72e-10
Gal sigma 0.25 rms: 6.78e-05, 9.88e-06, 5.03e-08, 4.32e-10, 4.31e-10

PSF sigma = 1.0
RMSes:
Gal sigma 2 rms: [3.63e-14, 1.71e-10, 1.71e-10, 1.71e-10, 1.71e-10]
Gal sigma 1 rms: [2.53e-07, 1.24e-07, 1.24e-07, 1.24e-07, 1.24e-07]
Gal sigma 0.5 rms: [0.000134, 1.13e-06, 1.13e-06, 1.13e-06, 1.13e-06]
Gal sigma 0.25 rms: [0.00021, 3.15e-05, 7.08e-07, 7.39e-07, 7.4e-07]




'''
