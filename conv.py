from __future__ import print_function
import scipy.stats
from pylab import *
import numpy as np

from astrometry.util.plotutils import PlotSequence, dimshow

from compare import measure
    
from demo import galaxy_psf_convolution, GaussianGalaxy

def integrate_gaussian(G, x, y):
    xx,yy = np.meshgrid(x, y)
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


def render_airy(params, x, y):
    scale,center = params
    airyx = np.abs((x-center) * scale)
    airyx = np.hypot(airyx[:,np.newaxis], airyx[np.newaxis,:])
    A = (2. * scipy.special.j1(airyx) / (airyx))**2
    A[airyx == 0] = 1.
    A /= np.sum(A)
    return A
    #A = (2. * scipy.special.j1(airyx) / (airyx))**2
    #A[airyx == 0] = 1.
    #pixpsf = A[:,np.newaxis] * A[np.newaxis,:]
    #return pixpsf

def compare_subsampled(S, s, ps, psf, pixpsf, Gmine,v,w, gal_sigma, psf_sigma,
                       cd,
                       get_ffts=False, eval_psf=integrate_gaussian):
    print()
    print('Subsample', s)
    print()

    step = 1./s
    sz = s * (S-1) + 1
    
    #x = np.arange(0, S, step)[:sz+1]
    x = np.arange(0, S, step)[:sz]
    #y = np.arange(0, S, step)[:sz+1]
    # Create pixelized PSF (Gaussian)
    sx = x - 0.5 + step/2.
    subpixpsf = eval_psf(psf, sx, sx)
    binned = bin_image(subpixpsf, s)

    bh,bw = binned.shape
    pixpsf1 = pixpsf[:bh,:bw]
    ph,pw = pixpsf.shape
    binned = binned[:ph,:pw]

    print('Binned PSF:')
    measure(binned)
    print('Pix PSF:')
    measure(pixpsf)

    # Recompute my convolution using the binned PSF
    P,FG,Gmine,v,w = galaxy_psf_convolution(
        gal_sigma, 0., 0., GaussianGalaxy, cd,
        0., 0., binned, debug=True)

    xx,yy = np.meshgrid(x,x)

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
    #gxx,gyy = xx - step, yy - step
    #gxx,gyy = xx - step/2., yy - step/2.
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
    # for i in range(len(w)):
    #     plt.plot(v, Fpsf[i,:], 'c-')
    # for i in range(len(v)):
    #     plt.plot(w, Fpsf[:,i], 'm-')
    # plt.title('PSF Fourier transform')
    # ps.savefig()
    # 
    # IV = np.argsort(v)
    # IW = np.argsort(w)
    # plt.clf()
    # for i in range(len(w)):
    #     plt.plot(v[IV], np.abs(Fpsf[i,IV]), 'c-')
    # for i in range(len(v)):
    #     plt.plot(w[IW], np.abs(Fpsf[IW,i]), 'm-')
    # plt.title('abs PSF Fourier transform')
    # ps.savefig()
    # 
    # plt.yscale('log')
    # ps.savefig()
    
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
    plt.title('log binned FFT conv')
    plt.colorbar()

    plt.subplot(2,3,5)
    dimshow(np.log10(np.maximum(Gmine / np.max(Gmine), 1e-12)))
    #dimshow(Gmine)
    plt.title('log my conv')
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

    #for psf_sigma in [2., 1.5, 1.]:
    for psf_sigma in [2.]:

        rms2 = []

        x = np.arange(S)
        y = np.arange(S)
        xx,yy = np.meshgrid(x, y)

        scale = 1.5 / psf_sigma
        pixpsf = render_airy((scale, center), x, y)
        psf = (scale,center)
        eval_psf = render_airy


        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(x, pixpsf[center,:], 'b-')
        plt.plot(x, pixpsf[:,center], 'r-')
        plt.subplot(2,1,2)
        plt.plot(x, np.maximum(1e-16, pixpsf[center,:]), 'b-')
        plt.plot(x, np.maximum(1e-16, pixpsf[:,center]), 'r-')
        plt.yscale('log')
        ps.savefig()

        plt.clf()
        plt.imshow(pixpsf, interpolation='nearest', origin='lower')
        ps.savefig()

        plt.clf()
        plt.imshow(np.log10(np.maximum(1e-16, pixpsf)),
                   interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.title('log PSF')
        ps.savefig()
        
        # psf
        #psf = scipy.stats.norm(loc=center + 0.5, scale=psf_sigma)

        # plt.clf()
        # plt.imshow(Pcdf, interpolation='nearest', origin='lower')
        # ps.savefig()

        # #Pcdf = psf.cdf(xx) * psf.cdf(yy)
        # #pixpsf = integrate_gaussian(psf, xx, yy)
        # 
        # padpsf = np.zeros((S*2-1, S*2-1))
        # ph,pw = pixpsf.shape
        # padpsf[S/2:S/2+ph, S/2:S/2+pw] = pixpsf
        # Fpsf = np.fft.rfft2(padpsf)
        # 
        # padh,padw = padpsf.shape
        # v = np.fft.rfftfreq(padw)
        # w = np.fft.fftfreq(padh)
        # fmax = max(max(np.abs(v)), max(np.abs(w)))
        # cut = fmax / 2. * 1.000001
        # #print('Frequence cut:', cut)
        # Ffiltpsf = Fpsf.copy()
        # #print('Ffiltpsf', Ffiltpsf.shape)
        # #print((np.abs(w) < cut).shape)
        # #print((np.abs(v) < cut).shape)
        # Ffiltpsf[np.abs(w) > cut, :] = 0.
        # Ffiltpsf[:, np.abs(v) > cut] = 0.
        # #print('pad v', v)
        # #print('pad w', w)
        # 
        # filtpsf = np.fft.irfft2(Ffiltpsf, s=(padh,padw))
        # 
        # print('filtered PSF real', np.max(np.abs(filtpsf.real)))
        # print('filtered PSF imag', np.max(np.abs(filtpsf.imag)))
        # 
        # plt.clf()
        # plt.subplot(2,3,1)
        # dimshow(Fpsf.real)
        # plt.colorbar()
        # plt.title('Padded PSF real')
        # plt.subplot(2,3,4)
        # dimshow(Fpsf.imag)
        # plt.colorbar()
        # plt.title('Padded PSF imag')
        # 
        # plt.subplot(2,3,2)
        # dimshow(Ffiltpsf.real)
        # plt.colorbar()
        # plt.title('Filt PSF real')
        # plt.subplot(2,3,5)
        # dimshow(Ffiltpsf.imag)
        # plt.colorbar()
        # plt.title('Filt PSF imag')
        # 
        # plt.subplot(2,3,3)
        # dimshow(filtpsf.real)
        # plt.title('PSF real')
        # plt.colorbar()
        # 
        # plt.subplot(2,3,6)
        # dimshow(filtpsf.imag)
        # plt.title('PSF imag')
        # plt.colorbar()
        # 
        # ps.savefig()
        # 
        # 
        # pixpsf = filtpsf
        
        
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

            Fpsf, Fgal = compare_subsampled(
                S, 1, ps, psf, pixpsf, Gmine,v,w,
                gal_sigma, psf_sigma, cd, get_ffts=True, eval_psf=eval_psf)
            
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
            
            subsample = [1,2,4]
            rms1 = []
            for s in subsample:
                rms = compare_subsampled(S, s, ps, psf, pixpsf, Gmine,v,w, gal_sigma, psf_sigma, cd, eval_psf=eval_psf)
                rms1.append(rms)
            rms2.append(rms1)


        print()
        print('PSF sigma =', psf_sigma)
        print('RMSes:')
        for rms1,gal_sigma in zip(rms2, gal_sigmas):
            print('Gal sigma', gal_sigma, 'rms:',
                  ', '.join(['%.3g' % r for r in rms1]))
                

if __name__ == '__main__':
    #main()

    from tractor.psfex import *

    psffn = '/Users/dstn/legacypipe-dir/calib/decam/psfex/00257/00257496/decam-00257496-N1.fits'
    print('Reading PsfEx model from', psffn)
    psf = PixelizedPsfEx(psffn)

    cpsf = psf.constantPsfAt(1000,1000)

    pixpsf = cpsf.img
    ph,pw = pixpsf.shape

    gal_re = 10.
    
    assert(ph == pw)
    
    ps = PlotSequence('conv')

    plt.clf()
    plt.imshow(pixpsf, interpolation='nearest', origin='lower')
    ps.savefig()
    
    Fpsf = np.fft.rfft2(pixpsf)
    
    plt.clf()
    plt.subplot(2,4,1)
    dimshow(Fpsf.real)
    plt.colorbar()
    plt.title('PSF real')
    plt.subplot(2,4,5)
    dimshow(Fpsf.imag)
    plt.colorbar()
    plt.title('PSF imag')
    ps.savefig()

    # Subsample the PSF via resampling
    from astrometry.util.util import lanczos_shift_image

    scale = 2
    sh,sw = ph*scale, pw*scale
    subpixpsf = np.zeros((sh,sw))
    for ix in np.arange(scale):
        for iy in np.arange(scale):
            dx = ix / float(scale)
            dy = iy / float(scale)
            if ix == 0 and iy == 0:
                subpixpsf[0::scale, 0::scale] = pixpsf
                continue
            shift = lanczos_shift_image(pixpsf, -dx, -dy, order=5)
            subpixpsf[iy::scale, ix::scale] = shift

    # Evaluate a R_e = 1 pixel deVauc on the native pixel grid, using
    # the Gaussian approximation
    xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
    center = pw/2
    r2 = (xx - center)**2 + (yy - center)**2
    pixgal = np.zeros_like(pixpsf)
    
    from demo import DevGalaxy
    gal = DevGalaxy()
    
    for a,v in zip(gal.amp, gal.var):
        vv = v * gal_re**2
        ## FIXME ??? prefactors?
        #pixgal += a * 1./np.sqrt(vv) * np.exp(-0.5 * r2 / vv)
        pixgal += a * 1./vv * np.exp(-0.5 * r2 / vv)

    subpixgal = np.zeros_like(subpixpsf)
    xx,yy = np.meshgrid(np.arange(sw), np.arange(sh))
    r2 = (xx/float(scale) - center)**2 + (yy/float(scale) - center)**2

    for a,v in zip(gal.amp, gal.var):
        ## FIXME ??? prefactors?
        vv = v * gal_re**2
        #subpixgal += a * 1./np.sqrt(vv) * np.exp(-0.5 * r2 / vv)
        subpixgal += a * 1./vv * np.exp(-0.5 * r2 / vv)

    plt.clf()
    plt.loglog(r2[sh/2,:]+1, subpixgal[sh/2,:], 'b-')
    ps.savefig()
    
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(pixpsf, interpolation='nearest', origin='lower')
    plt.subplot(2,2,2)
    plt.imshow(subpixpsf, interpolation='nearest', origin='lower')

    plt.subplot(2,2,3)
    plt.imshow(pixgal, interpolation='nearest', origin='lower')
    plt.subplot(2,2,4)
    plt.imshow(subpixgal, interpolation='nearest', origin='lower')

    ps.savefig()
            
    Fsub = np.fft.rfft2(subpixpsf)

    Fgal = np.fft.rfft2(np.fft.ifftshift(pixgal))
    Fsubgal = np.fft.rfft2(np.fft.ifftshift(subpixgal))
    
    plt.clf()
    plt.subplot(2,4,1)
    dimshow(Fpsf.real)
    plt.colorbar()
    plt.title('PSF real')
    plt.subplot(2,4,5)
    dimshow(Fpsf.imag)
    plt.colorbar()
    plt.title('PSF imag')

    plt.subplot(2,4,2)
    dimshow(Fsub.real)
    plt.colorbar()
    plt.title('SubPSF real')
    plt.subplot(2,4,6)
    dimshow(Fsub.imag)
    plt.colorbar()
    plt.title('SubPSF imag')

    plt.subplot(2,4,3)
    dimshow(Fgal.real)
    plt.colorbar()
    plt.title('pix Gal real')
    plt.subplot(2,4,7)
    dimshow(Fgal.imag)
    plt.colorbar()
    plt.title('pix Gal imag')

    plt.subplot(2,4,4)
    dimshow(Fsubgal.real)
    plt.colorbar()
    plt.title('subpix Gal real')
    plt.subplot(2,4,8)
    dimshow(Fsubgal.imag)
    plt.colorbar()
    plt.title('subpix Gal imag')
    
    ps.savefig()


    pixconv = np.fft.irfft2(Fpsf * Fgal, s=pixpsf.shape)
    subpixconv = np.fft.irfft2(Fsub * Fsubgal, s=subpixpsf.shape)

    # my convolution
    pixscale = 1.
    cd = pixscale * np.eye(2) / 3600.
    Gmine = galaxy_psf_convolution(gal_re, 0., 0., gal, cd,
                                   0., 0., pixpsf)

    pixconv /= np.sum(pixconv)
    subpixconv /= np.sum(subpixconv)
    subpixconv *= scale**2
    Gmine /= np.sum(Gmine)
    
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(pixconv, interpolation='nearest', origin='lower')
    plt.subplot(1,3,2)
    plt.imshow(subpixconv, interpolation='nearest', origin='lower')
    plt.subplot(1,3,3)
    plt.imshow(Gmine, interpolation='nearest', origin='lower')
    ps.savefig()    

    plt.clf()
    plt.plot(np.arange(pw), pixconv[ph/2,:], 'b-')
    plt.plot(np.arange(sw)/float(scale), subpixconv[sh/2,:], 'r-')
    plt.plot(np.arange(pw), Gmine[ph/2,:], 'k-')
    plt.plot(np.arange(pw), pixpsf[ph/2,:], 'g-')
    ps.savefig()    

    plt.yscale('log')
    ps.savefig()    
    

    
    

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
