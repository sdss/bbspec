import numpy as n
import pyfits as pf
from bbspec.spec2d import boltonfuncs as GH
from scipy import special,interpolate,where,linalg

class arcmodel2D:

    x = n.arange(-5,6,1) 
    y = n.arange(-5,6,1)
    fibNo = 500
    nwavelen = 65
    fibBun = 20
    xpoints = 4114
    ypoints = 4128
    degree =  3
    nbund =  25
    yvalues = n.arange(0,ypoints,1)
    #bStart = 2 # starting bundle
    #bEnd = 2 # ending bundle

    def __init__(self,indir = '.', outdir = '.'):
        self.indir = indir
        self.outdir = outdir
    
    def setarc_flat(self,arcid,flatid):
        self.arcid = arcid
        self.flatid = flatid
        
        spArc_file = self.indir + '/spArc-' + self.arcid + '.fits.gz' 
        spFlat_file = self.indir + '/spFlat-' + self.flatid + '.fits.gz' 
        data_file = self.indir + '/sdProc-' + self.arcid + '.fits' 
        
        self.h_spArc = pf.open(spArc_file)
        self.h_spFlat = pf.open(spFlat_file)
        self.data = pf.open(data_file)

    def close(self):
        self.h_spArc.close()
        self.h_spFlat.close()
        self.data.close()

    def model_arc(self, bStart, bEnd):
        bStart = int(bStart)
        bEnd = int(bEnd)
        loopBund =  bEnd - bStart + 1
        #- make floating point errors fatal (fail early, fail often)
        n.seterr(all='raise')

        # Data & invvar from sdR files
        biasSubImg = self.data[0].data
        invvr = self.data[1].data

        # Data from spArc files
        [goodwaveypos, good_wavelength, wavelength, arcSigma] = self.dataspArc()

        # Data from spFlat files
        [fiberflat, xpos_final, flatSigma] = self.dataspFlat()

        # allocating space for variables
        arcmodel2D.yvalues = n.arange(0,arcmodel2D.ypoints,1)
        GHparam = n.zeros((arcmodel2D.nwavelen,arcmodel2D.nbund,15,15))
        fib = n.arange(0,arcmodel2D.fibNo,1) 
        ngoodwave=len(good_wavelength)
        chi_sqrd = n.zeros((arcmodel2D.nwavelen,arcmodel2D.nbund,1))

        coeffAll  = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree + 1,15,15))
        xcenter  = n.zeros((arcmodel2D.nwavelen,arcmodel2D.nbund,arcmodel2D.fibBun))
        ycenter  = n.zeros((arcmodel2D.nwavelen,arcmodel2D.nbund,arcmodel2D.fibBun))
        sigma  = n.zeros((arcmodel2D.nwavelen,arcmodel2D.nbund,arcmodel2D.fibBun))

        # define max order
        maxorder = 4	

        for i_bund in range(bStart, bEnd+1):
            print i_bund
            kcons = i_bund*arcmodel2D.fibBun
            for i_actwave in range(0, arcmodel2D.nwavelen):
                # declare variables
                actwavelength = good_wavelength[i_actwave]
                func = interpolate.interp1d(good_wavelength,goodwaveypos[:,fib[kcons]], kind='cubic')
                yact = func(actwavelength)
                y1 = round(yact) -10 
                y2 = round(yact) +10                                                                                                 
                x1 = round(xpos_final[round(yact),fib[kcons]])
                x2 = round(xpos_final[round(yact),fib[kcons+19]])
                x1im = (x1)-10
                x2im = (x2)+10
                y1im = (y1)-10
                y2im = (y2)+10
                fibcons = n.arange(kcons,kcons+arcmodel2D.fibBun,1)	
                rowimage = biasSubImg[y1im:y2im+1,x1im:x2im+1]
                x_pix = n.arange(x1im,x2im+1,1)  
                y_pix  = n.arange(y1im,y2im+1,1)
                numx = len(x_pix)  
                numy = len(y_pix)
                numk = len(fibcons) 
                numlambda_val = 1
            flag = 0	 
            n_k = n.zeros((len(fibcons),1))
            basisfuncstack = n.zeros((numlambda_val,numk,numy,numx))
            basisstack = n.zeros((numy*numx,1))
            mm = 0 ; nn = 0
            # Creating basis function
            for mor in range (0,maxorder+1):
                for nor in range (0, maxorder+1):
                    if (mor+nor <= 4):
                        mm = n.hstack((mm,mor))
                        nn = n.hstack((nn,nor))
                        [ff, basisfunc, basisimage,flag,xcenarr,ycenarr, sigmaarr] = self.create_basisfunc(actwavelength,good_wavelength,goodwaveypos,xpos_final,arcSigma,flatSigma,rowimage,x1im,x2im,y1im,y2im,fiberflat,kcons,mor,nor) 	
                        # checking if all values are non-zero
                        if (flag == 1):			
                            basisfuncstack = n.hstack((basisfuncstack,basisfunc))
                            basisstack = n.hstack((basisstack,basisimage))
                        else:
                            break 
                if (flag == 0):		
                    break
            if (flag == 0):	
                continue
            

            basis = n.sum(n.sum(basisfuncstack[:,:,:,:],axis = 1),axis = 0)
            nk = n.shape(basisfunc)[1]
            nlambda = n.shape(basisfunc)[0]
            ni = n.shape(basisfunc)[2]
            nj = n.shape(basisfunc)[3]

            N = n.zeros((ni*nj,ni*nj))
            invvr_sub = invvr[y1im:y2im+1,x1im:x2im+1]		
            flat_invvr = n.ravel(invvr_sub)
            N = n.diag(flat_invvr)
            
            p = n.ravel(rowimage)
            p1= n.zeros((len(p),1))
            p1[:,0] = p
            B1 = basisstack[:,1:]
            scaledbasis = n.zeros((ni*nj ,1))
            [theta,t2,l1,t1,t2] = self.calparam(B1,N,p1)
            GHparam[i_actwave,i_bund,mm[1:],nn[1:]] = theta[:,0]/theta[0,0]
                
                # model image
            scaledbasis = n.dot(B1, theta)
            scaledbasis1 = n.zeros((len(scaledbasis[:,0]),1))
            scaledbasis1[:,0] = scaledbasis[:,0]
            
            #chi-squared value
            ndiag = n.zeros((len(N),1))
            ndiag[:,0] = n.diag(N)
            nz = where(scaledbasis1 != 0)
            err = (p1[nz]-scaledbasis1[nz])*n.sqrt(ndiag[nz])
            chi_sqrd[i_actwave,i_bund,0] = n.mean(n.power(err,2))
            scaledbasis.resize(ni,nj)                                                                                                                        
            scaledbasis4 = scaledbasis
            
            # Residual values	
            residualGH = rowimage - scaledbasis
        
            xcenter[i_actwave,i_bund,:]  = xcenarr
            ycenter[i_actwave,i_bund,:]  = ycenarr
            sigma[i_actwave,i_bund,:]  = sigmaarr 
            
            #arcmodel2D.degree = 3
            
            # wavelength indexes which do not have outliers 
            reqwave = n.array([11,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 32, 33, 34,  41, 44, 46, 47, 48, 49, 50, 52, 53,55, 56, 57, 59, 60,62])

        # loop to get the value of PSF parameters at all wavelengths		    
            for i_fib in range(0, arcmodel2D.fibBun):
                for i_plot in range(1, len(mm)):
                    wavecons   =  wavelength[:, i_bund*arcmodel2D.fibBun + i_fib]
                    param = GHparam[reqwave, i_bund, mm[i_plot], nn[i_plot]]
                    z = n.polyfit(good_wavelength[reqwave], param, arcmodel2D.degree)
                    p = n.poly1d(z)
                    coeffAll[i_fib, :, mm[i_plot], nn[i_plot]] = z
        
        PSFArc = self.createPSFArc(GHparam,xcenter, ycenter, sigma,good_wavelength,mm,nn,bStart)
        PSFBasis = self.createPSFBasis(coeffAll, wavelength, xpos_final, flatSigma,good_wavelength,mm,nn,bStart)
        return (PSFArc , PSFBasis)

    # Function creates basis function at known wavelengths of arc-frames
    def create_basisfunc(self, actwavelength,good_wavelength,goodwaveypos,xpos_final,arcSigma,flatSigma,rowimage,x1im,x2im,y1im,y2im,fiberflat,kcons,mor,nor):
        datacenterval = n.zeros((arcmodel2D.fibBun,1))
        fibcons = n.arange(kcons,kcons+arcmodel2D.fibBun,1) 
        n_k = n.zeros((len(fibcons),1))
        numx = n.shape(rowimage)[1]
        numy = n.shape(rowimage)[0]
        numk = len(fibcons) 
        numlambda_val = 1		
        basisfunc = n.zeros((numlambda_val,numk,numy,numx)) 
        zero1 = n.zeros((numlambda_val,numk,numy,numx)) 
        zero2 = n.zeros((numy*numx,1)) 	
        basisstack = 0
        xcenarr = n.zeros((arcmodel2D.fibBun))	
        ycenarr = n.zeros((arcmodel2D.fibBun))	
        sigmaarr = n.zeros((arcmodel2D.fibBun))	
        for i_k in range(0,numk):
            testwave = actwavelength
            func = interpolate.interp1d(good_wavelength,goodwaveypos[:,fibcons[i_k]], kind='cubic')
            ypix = func(testwave)
            xpix = xpos_final[round(ypix),fibcons[i_k]]
            sigmaA = arcSigma[round(ypix),fibcons[i_k]] 
            sigmaF = flatSigma[round(ypix),fibcons[i_k]]
            iceny = round(ypix)
            ycen  = ypix
            yr = ycen-iceny
            cenA = yr
            icenx = round(xpix)
            xcen  = (xpix)
            xr = xcen-icenx
            cenF = xr        
            xcenarr[i_k] = xcen
            ycenarr[i_k] = ycen
            sigmaarr[i_k] = sigmaF
            starty= iceny-y1im
            startx= icenx-x1im
            xPSF = n.zeros((len(arcmodel2D.x),1))
            yPSF = n.zeros((len(arcmodel2D.y),1))
            #xPSF[:,0] = GH.pgh(x=arcmodel2D.x, xc = cenF, sigma = sigmaF ,m = mor)
            xPSF[:,0] = GH.pgh(arcmodel2D.x, mor, cenF, sigmaF)
            yPSF[:,0] = GH.pgh(arcmodel2D.y, nor, cenA, sigmaF)
            out = n.outer(yPSF,xPSF)
            datacenterval[i_k,0] = rowimage[starty,startx]
            basisfunc[0,i_k,starty+arcmodel2D.y[0]:(starty+arcmodel2D.y[len(arcmodel2D.y)-1]+1),startx+arcmodel2D.x[0]:(startx+arcmodel2D.x[len(arcmodel2D.x)-1]+1)] = out
            func_n_k = interpolate.interp1d(arcmodel2D.yvalues,fiberflat[fibcons[i_k],:])
            n_kVal = func_n_k(ypix)
            # relative fiber throughput (n_k) taken as 1
            n_k[i_k,0] = 1
        #n_kzero = where(n_k[:,0] == 0)
        #   n_kzero = 0 	
        #if (shape(n_kzero)[1] > 0 ):
        #	flag = 0
        #	return(n_k,zero1,zero2,flag,xcenarr,ycenarr,sigmaarr)
        #else:	
        basisimage = self.basisimg(basisfunc, n_k)
        flag = 1		
        return (n_k,basisfunc, basisimage,flag,xcenarr, ycenarr, sigmaarr) 	

    # supress the four dimensional basis function to two-dimensional
    def basisimg(self, A00, n_k):
        nlambda = n.shape(A00)[0]
        nk = n.shape(A00)[1]
        ni = n.shape(A00)[2]
        nj = n.shape(A00)[3]
        A_twod = n.zeros((ni*nj,nlambda*nk))
        B00= n.zeros((ni*nj ,1))
        i_l = 0
        i_m = 0
        numlambda_val = nlambda
        numk = nk
        for i_lam in range(0,numlambda_val):
            for i_k in range(0,numk):
                a =  A00[i_lam, i_k,:,:]
                b = n.ravel(a)
                nterms = len(b)
                A_twod[:,i_m]  =  b
                i_m = i_m + 1
        B00 = n.dot(A_twod,n_k) 			
        return B00

    def calparam(self, B,N,p1):
        l1 = n.dot(n.transpose(B),N)
        t1 = n.dot(l1,p1)
        t2 = n.dot(l1,B)
        theta = n.dot(linalg.pinv(t2),t1)
        return(theta,t2,l1,t1,t2)
        
    # write to FITS file
    def createPSFBasis(self, coeffAll, wavelength, xpos_final, flatSigma,good_wavelength,mm,nn, bStart):
        theta0 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta1 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta2 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta3 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta4 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta5 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta6 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta7 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta8 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta9 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta10 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta11 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta12 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta13 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        theta14 = n.zeros((arcmodel2D.fibBun,arcmodel2D.degree+1))
        final_wavelength  = n.zeros((arcmodel2D.fibBun,arcmodel2D.ypoints))

        theta0  = coeffAll[: , :,mm[1],nn[1]]
        theta1   = coeffAll[: , :,mm[2],nn[2]]
        theta2 =  coeffAll[: , :,mm[3],nn[3]]
        theta3 =  coeffAll[: ,:, mm[4],nn[4]]
        theta4 =  coeffAll[: , :,mm[5],nn[5]]
        theta5  =  coeffAll[: , :,mm[6],nn[6]]
        theta6 =  coeffAll[: , :,mm[7],nn[7]]
        theta7 =  coeffAll[: , :,mm[8],nn[8]]
        theta8 =  coeffAll[: , :,mm[9],nn[9]]
        theta9 =  coeffAll[: , :,mm[10],nn[10]]
        theta10 =  coeffAll[: , :,mm[11],nn[11]]
        theta11 =  coeffAll[: ,:, mm[12],nn[12]]
        theta12 =  coeffAll[: ,:, mm[13],nn[13]]
        theta13 =  coeffAll[: , :,mm[14],nn[14]]
        theta14 =  coeffAll[: , :,mm[15],nn[15]]
            
        a = n.arange(0,arcmodel2D.ypoints, dtype=float)
        b=n.transpose(a)
        q = b.repeat(arcmodel2D.fibBun)
        a = n.reshape(q, (arcmodel2D.ypoints,arcmodel2D.fibBun))
        ycenterf  = n.transpose(a)
        
        xcenterf = n.transpose(xpos_final[:, bStart*arcmodel2D.fibBun : bStart*arcmodel2D.fibBun+arcmodel2D.fibBun])
        sigmaarrf = n.transpose(flatSigma[:, bStart*arcmodel2D.fibBun : bStart*arcmodel2D.fibBun+arcmodel2D.fibBun])	
        final_wavelength= n.transpose(n.log10(wavelength[:, bStart*arcmodel2D.fibBun : bStart*arcmodel2D.fibBun+arcmodel2D.fibBun]))

        hdu0 = pf.PrimaryHDU(xcenterf)
        hdu0.header.update('PSFTYPE', 'GAUSS-HERMITE', 'GAUSS-HERMITE POLYNOMIALS') 
        hdu0.header.update('NPIX_X', arcmodel2D.xpoints, 'number of image pixels in the X-direction')
        hdu0.header.update('NPIX_Y', arcmodel2D.ypoints, 'number of image pixels in the Y-direction')
        hdu0.header.update('NFLUX',  arcmodel2D.ypoints,   'number of flux bins per spectrum [NAXIS1]')
        hdu0.header.update('NSPEC',  arcmodel2D.fibBun,  'number of spectra [NAXIS2]')
        hdu0.header.update('PSFPARAM', 'X', 'X position as a function of flux bin')

        hdu1 = pf.ImageHDU(ycenterf)
        hdu1.header.update('PSFPARAM', 'Y', 'Y position as a function of flux bin')

        hdu2 = pf.ImageHDU(final_wavelength)
        hdu2.header.update('PSFPARAM', 'LogLam', 'Known Wavelengths of an arc-frame')

        hdu3 = pf.ImageHDU(sigmaarrf)
        hdu3.header.update('PSFPARAM', 'sigma', 'Gaussian sigma values of basis')

        hdu4 = pf.ImageHDU(theta0)
        hdu4.header.update('PSFPARAM', 'PGH(0,0)', 'Pixelated Gauss-Hermite Order: (0,0)')

        hdu5 = pf.ImageHDU(theta1)
        hdu5.header.update('PSFPARAM', 'PGH(0,1)', 'Pixelated Gauss-Hermite Order: (0,1)')

        hdu6 = pf.ImageHDU(theta2)
        hdu6.header.update('PSFPARAM', 'PGH(0,2)', 'Pixelated Gauss-Hermite Order: (0,2)')

        hdu7 = pf.ImageHDU(theta3)
        hdu7.header.update('PSFPARAM', 'PGH(0,3)', 'Pixelated Gauss-Hermite Order: (0,3)')

        hdu8 = pf.ImageHDU(theta4)
        hdu8.header.update('PSFPARAM', 'PGH(0,4)', 'Pixelated Gauss-Hermite Order: (0,4)') 

        hdu9 = pf.ImageHDU(theta5)
        hdu9.header.update('PSFPARAM', 'PGH(1,0)', 'Pixelated Gauss-Hermite Order: (1,0)')

        hdu10 = pf.ImageHDU(theta6)
        hdu10.header.update('PSFPARAM', 'PGH(1,1)', 'Pixelated Gauss-Hermite Order: (1,1)')

        hdu11 = pf.ImageHDU(theta7)
        hdu11.header.update('PSFPARAM', 'PGH(1,2)', 'Pixelated Gauss-Hermite Order: (1,2)')

        hdu12 = pf.ImageHDU(theta8)
        hdu12.header.update('PSFPARAM', 'PGH(1,3)', 'Pixelated Gauss-Hermite Order: (1,3)')

        hdu13 = pf.ImageHDU(theta9)
        hdu13.header.update('PSFPARAM', 'PGH(2,0)', 'Pixelated Gauss-Hermite Order: (2,0)')

        hdu14 = pf.ImageHDU(theta10)
        hdu14.header.update('PSFPARAM', 'PGH(2,1)', 'Pixelated Gauss-Hermite Order: (2,1)')

        hdu15 = pf.ImageHDU(theta11)
        hdu15.header.update('PSFPARAM', 'PGH(2,2)', 'Pixelated Gauss-Hermite Order: (2,2)')

        hdu16 = pf.ImageHDU(theta12)
        hdu16.header.update('PSFPARAM', 'PGH(3,0)', 'Pixelated Gauss-Hermite Order: (3,0)')

        hdu17 = pf.ImageHDU(theta13)
        hdu17.header.update('PSFPARAM', 'PGH(3,1)', 'Pixelated Gauss-Hermite Order: (3,1)')

        hdu18 = pf.ImageHDU(theta14)
        hdu18.header.update('PSFPARAM', 'PGH(4,0)', 'Pixelated Gauss-Hermite Order: (4,0)')
        hdulist = pf.HDUList([hdu0, hdu1, hdu2,hdu3,hdu4,hdu5,hdu6,hdu7,hdu8,hdu9,hdu10,hdu11,hdu12,hdu13,hdu14, hdu15, hdu16, hdu17, hdu18])
        fname = self.outdir + '/spBasisPSF-' + self.arcid+'.fits'
        #fname = 'demo1.fits'
        #print 'reached basis'
        hdulist.writeto(fname, clobber=True)
        return (fname)

    def createPSFArc(self, GHparam, xcenter, ycenter, sigma,good_wavelength,mm,nn, bStart):
        xcenterf  = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        ycenterf  = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        sigmaarrf  = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta0 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta1 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta2 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta3 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta4 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta5 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta6 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta7 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta8 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta9 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta10 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta11 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta12 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta13 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        theta14 = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))
        final_wavelength  = n.zeros((arcmodel2D.fibNo,arcmodel2D.nwavelen))

        for i_wave in range(0, arcmodel2D.nwavelen):
            theta0[:,i_wave]  = GHparam[i_wave, : , mm[1],nn[1]].repeat(arcmodel2D.fibBun)
            theta1[:,i_wave]   = GHparam[i_wave, : , mm[2],nn[2]].repeat(arcmodel2D.fibBun)
            theta2[:,i_wave] =  GHparam[i_wave, : , mm[3],nn[3]].repeat(arcmodel2D.fibBun)
            theta3[:,i_wave] =  GHparam[i_wave, : , mm[4],nn[4]].repeat(arcmodel2D.fibBun)
            theta4[:,i_wave] =  GHparam[i_wave, : , mm[5],nn[5]].repeat(arcmodel2D.fibBun)
            theta5[:,i_wave]  =  GHparam[i_wave, : , mm[6],nn[6]].repeat(arcmodel2D.fibBun)
            theta6[:,i_wave] =  GHparam[i_wave, : , mm[7],nn[7]].repeat(arcmodel2D.fibBun)
            theta7[:,i_wave] =  GHparam[i_wave, : , mm[8],nn[8]].repeat(arcmodel2D.fibBun)
            theta8[:,i_wave] =  GHparam[i_wave, : , mm[9],nn[9]].repeat(arcmodel2D.fibBun)
            theta9[:,i_wave] =  GHparam[i_wave, : , mm[10],nn[10]].repeat(arcmodel2D.fibBun)
            theta10[:,i_wave] =  GHparam[i_wave, : , mm[11],nn[11]].repeat(arcmodel2D.fibBun)
            theta11[:,i_wave] =  GHparam[i_wave, : , mm[12],nn[12]].repeat(arcmodel2D.fibBun)
            theta12[:,i_wave] =  GHparam[i_wave, : , mm[13],nn[13]].repeat(arcmodel2D.fibBun)
            theta13[:,i_wave] =  GHparam[i_wave, : , mm[14],nn[14]].repeat(arcmodel2D.fibBun)
            theta14[:,i_wave] =  GHparam[i_wave, : , mm[15],nn[15]].repeat(arcmodel2D.fibBun)
            
            xcenterf[:,i_wave]  = n.ravel(xcenter[i_wave, :, :])
            ycenterf[:,i_wave]  = n.ravel(ycenter[i_wave, :, :])
            
            final_wavelength[:,i_wave] = (n.log10(good_wavelength[i_wave])).repeat(arcmodel2D.fibNo)
            
            sigmaarrf[:,i_wave]  = n.ravel(sigma[i_wave, :, :])

            
        # write to FITS file
        hdu0 = pf.PrimaryHDU(xcenterf)	
        hdu0.header.update('PSFTYPE', 'GAUSS-HERMITE', 'GAUSS-HERMITE POLYNOMIALS') 
        hdu0.header.update('NPIX_X', arcmodel2D.xpoints, 'number of image pixels in the X-direction')
        hdu0.header.update('NPIX_Y', arcmodel2D.ypoints, 'number of image pixels in the Y-direction')
        hdu0.header.update('NFLUX',  arcmodel2D.nwavelen,   'number of flux bins per spectrum [NAXIS1]')
        hdu0.header.update('NSPEC',  arcmodel2D.fibNo,  'number of spectra [NAXIS2]')
        hdu0.header.update('PSFPARAM', 'X', 'X position as a function of flux bin')

        hdu1 = pf.ImageHDU(ycenterf)
        hdu1.header.update('PSFPARAM', 'Y', 'Y position as a function of flux bin')

        hdu2 = pf.ImageHDU(final_wavelength)
        hdu2.header.update('PSFPARAM', 'LogLam', 'Known Wavelengths of an arc-frame')

        hdu3 = pf.ImageHDU(sigmaarrf)
        hdu3.header.update('PSFPARAM', 'sigma', 'Gaussian sigma values of basis')

        hdu4 = pf.ImageHDU(theta0)
        hdu4.header.update('PSFPARAM', 'PGH(0,0)', 'Pixelated Gauss-Hermite Order: (0,0)')

        hdu5 = pf.ImageHDU(theta1)
        hdu5.header.update('PSFPARAM', 'PGH(0,1)', 'Gauss-Hermite Order: (0,1)')

        hdu6 = pf.ImageHDU(theta2)
        hdu6.header.update('PSFPARAM', 'PGH(0,2)', 'Gauss-Hermite Order: (0,2)')

        hdu7 = pf.ImageHDU(theta3)
        hdu7.header.update('PSFPARAM', 'PGH(0,3)', 'Gauss-Hermite Order: (0,3)')

        hdu8 = pf.ImageHDU(theta4)
        hdu8.header.update('PSFPARAM', 'PGH(0,4)', 'Gauss-Hermite Order: (0,4)') 

        hdu9 = pf.ImageHDU(theta5)
        hdu9.header.update('PSFPARAM', 'PGH(1,0)', 'Gauss-Hermite Order: (1,0)')

        hdu10 = pf.ImageHDU(theta6)
        hdu10.header.update('PSFPARAM', 'PGH(1,1)', 'Gauss-Hermite Order: (1,1)')

        hdu11 = pf.ImageHDU(theta7)
        hdu11.header.update('PSFPARAM', 'PGH(1,2)', 'Gauss-Hermite Order: (1,2)')

        hdu12 = pf.ImageHDU(theta8)
        hdu12.header.update('PSFPARAM', 'PGH(1,3)', 'Gauss-Hermite Order: (1,3)')

        hdu13 = pf.ImageHDU(theta9)
        hdu13.header.update('PSFPARAM', 'PGH(2,0)', 'Gauss-Hermite Order: (2,0)')

        hdu14 = pf.ImageHDU(theta10)
        hdu14.header.update('PSFPARAM', 'PGH(2,1)', 'Gauss-Hermite Order: (2,1)')

        hdu15 = pf.ImageHDU(theta11)
        hdu15.header.update('PSFPARAM', 'PGH(2,2)', 'Gauss-Hermite Order: (2,2)')

        hdu16 = pf.ImageHDU(theta12)
        hdu16.header.update('PSFPARAM', 'PGH(3,0)', 'Gauss-Hermite Order: (3,0)')

        hdu17 = pf.ImageHDU(theta13)
        hdu17.header.update('PSFPARAM', 'PGH(3,1)', 'Gauss-Hermite Order: (3,1)')

        hdu18 = pf.ImageHDU(theta14)
        hdu18.header.update('PSFPARAM', 'PGH(4,0)', 'Gauss-Hermite Order: (4,0)')

        hdulist = pf.HDUList([hdu0, hdu1, hdu2,hdu3,hdu4,hdu5,hdu6,hdu7,hdu8,hdu9,hdu10,hdu11,hdu12,hdu13,hdu14, hdu15, hdu16, hdu17, hdu18])

        fname = self.outdir + '/spArcPSF-' + self.arcid +'.fits'
        #fname = 'demo2.fits'
        #hdulist.writeto(fname, clobber=True)
        #print 'reached Arcfile'
        return (fname)
        
    def dataspArc(self):

        # Number of fibers 
        image = self.h_spArc[0].data
        nfib = n.shape(image)[0]
        ndatapt = n.shape(image)[1]

        # y-sigma values
        ysigma = self.h_spArc[4].data
        xmin =  ysigma.field(1)  
        xmax =  ysigma.field(2)
        coeff = ysigma.field(3)
        
        #- Work around pyfits bug which doesn't properly support 2D arrays in tables
        ncoeff = coeff.size / nfib
        coeff = coeff.reshape( (nfib, ncoeff) )
        
        nx = xmax-xmin+1
        xbase = n.arange(xmin,xmax-xmin+1,1)
        x_1 = xmin
        x_2 = xmax
        a = 2/(xmax-xmin)
        b = -1-a*xmin
        x_p_base = a*xbase + b
        l= n.zeros((ndatapt,))
        legendreDim = n.shape(coeff)[1]
        # Constructing Legendre Polynomial
        for i_leg in range (0, legendreDim):
            ltemp =  special.legendre(i_leg)
            ltemp = ltemp(x_p_base)
            l = n.vstack((l,ltemp))

        ### r = n.transpose(coeff[0,:,:])
        r = coeff
        
        Ysig = n.dot(r,l[1:])
        Ysig = n.transpose(Ysig)
        arcSigma =Ysig
        
        #wavelength value at each Y-center
        waveset 	= self.h_spArc[2].data
        xminwave 	= waveset.field(1)
        xmaxwave 	= waveset.field(2)
        coeffwave 	= waveset.field(3)

        #- Work around pyfits bug which doesn't properly support 2D arrays in tables
        ncoeffwave = coeffwave.size / nfib
        coeffwave = coeffwave.reshape( (nfib, ncoeffwave) )

        nxwave = xmaxwave-xminwave+1
        xbasewave = n.arange(xminwave,xmaxwave-xminwave+1,1)
        x_1wave = xminwave
        x_2wave = xmaxwave
        awave = 2/(xmaxwave-xminwave)
        bwave = -1-awave*xminwave
        x_p_basewave = awave*xbasewave + bwave
        dimwavept = n.shape(coeffwave)[1]
        dimwavept = dimwavept/nfib
        lwave= n.zeros((ndatapt,))
        dimwavept = n.shape(coeffwave)[1]
        # Constructing Legendre Polynomial
        for i_leg in range (0, dimwavept):
            lwavetemp =  special.legendre(i_leg)
            lwavetemp = lwavetemp(x_p_basewave)
            lwave = n.vstack((lwave,lwavetemp))

        rwave = coeffwave.reshape(nfib,dimwavept)
        wavelength_tmp = n.dot(rwave,lwave[1:])
        wavelength= n.transpose(wavelength_tmp)
        wavelength = n.power(10,wavelength)
        
        # good wavelengths of Arc lamps 
        good_lambda_val = self.h_spArc[1].data
        good_wavelength = good_lambda_val[:,0]
        wavenum = n.shape(good_lambda_val)[0]
        lambdadim2 = n.shape(good_lambda_val)[1]
        goodwaveypos = good_lambda_val[0:wavenum, 1:lambdadim2]
        return(goodwaveypos, good_wavelength, wavelength, arcSigma)

    def dataspFlat(self):
        
        # Number of fibers 
        image = self.h_spFlat[0].data
        nfib = n.shape(image)[0]
        ndatapt = n.shape(image)[1] 
        
        # X- sigma 
        sigma	=  self.h_spFlat[3].data
        xminSig 	=  sigma.field(1)
        xmaxSig 	=  sigma.field(2)
        coeffXSig 	=  sigma.field(3)
        
        #- Work around pyfits bug which doesn't properly support 2D arrays in tables
        ncoeffXSig = coeffXSig.size / nfib
        coeffXSig = coeffXSig.reshape( (nfib, ncoeffXSig) )
        
        nxSig = xmaxSig-xminSig+1
        xbaseSig = n.arange(xminSig,xmaxSig-xminSig+1,1)
        x_1Sig = xminSig
        x_2Sig = xmaxSig
        aSig = 2/(xmaxSig-xminSig)
        bSig = -1-aSig*xminSig
        x_p_baseSig = aSig*xbaseSig + bSig
        dimXSig = n.shape(coeffXSig)[1]
        dimXSig = dimXSig/nfib
        lXSig= n.zeros((ndatapt,))
        dimXSig = n.shape(coeffXSig)[1]
        # Constructing Legendre Polynomial
        for i_leg in range (0, dimXSig):
            lSigtemp =  special.legendre(i_leg)
            lSigtemp = lSigtemp(x_p_baseSig)
            lXSig = n.vstack((lXSig,lSigtemp))
        
        rsigma = coeffXSig.reshape(nfib, dimXSig)
        Xsig = n.dot(rsigma,lXSig[1:])
        Xsig = n.transpose(Xsig)
        flatSigma = Xsig

        # X-centers
        peakPos = self.h_spFlat[1].data
        xmin =  peakPos.field(1)
        xmax =  peakPos.field(2)
        coeffxpos = peakPos.field(3)
        
        #- Work around pyfits bug which doesn't properly support 2D arrays in tables
        ncoeffxpos = coeffxpos.size / nfib
        coeffxpos = coeffxpos.reshape( (nfib, ncoeffxpos) )
        
        nx = xmax-xmin+1
        xbase = n.arange(xmin,xmax-xmin+1,1)
        x_1 = xmin
        x_2 = xmax
        a = 2/(xmax-xmin)
        b = -1-a*xmin
        x_p_base = a*xbase + b
        dimxpos = n.shape(coeffxpos)[1]
        dimxpos = dimxpos/nfib
        lxpos = n.zeros((ndatapt,))
        dimxpos  = n.shape(coeffxpos)[1]
        # Constructing Legendre Polynomial
        for i_leg in range (0, dimxpos):
            lxpostemp =  special.legendre(i_leg)
            lxpostemp = lxpostemp(x_p_base)
            lxpos  = n.vstack((lxpos ,lxpostemp))

        rxpos = coeffxpos.reshape(nfib, dimxpos)
        xpos = n.dot(rxpos,lxpos[1:])
        xpos_final = n.transpose(xpos)


        #fiber to fiber variatons from fiberflats
        fiberflat = self.h_spFlat[0].data
        return(fiberflat, xpos_final, flatSigma)
