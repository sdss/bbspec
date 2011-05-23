import numpy as n
import matplotlib as m
m.rc('text', usetex=True)
m.interactive(True)
from matplotlib import pyplot as p
from matplotlib import cm
import pyfits as pf
from scipy import optimize as opt
import  pylab as pp
from scipy import special
from scipy.sparse import *
from scipy import *
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import dsolve
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import linalg as sla
import scipy.special as spc
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import linalg 
from bbspec.spec2d import pgh as GH

x1 = -5
x2 =  6
y1 = -5
y2 =  6
x = n.arange(x1,x2,1) 
y = n.arange(y1,y2,1)
fibNo =  500
nwavelen = 65
fibBun = 20
ypoints = 4128
nbund =  25
yvalues = n.arange(0,ypoints,1)

def model_arc(arcid, flatid, indir = '.', outdir = '.'):

	spArc_file = indir + 'spArc-' + arcid + '.fits' 
	spFlat_file = indir + 'spFlat-' + flatid + '.fits' 
	data_file = indir + 'sdProc-' + arcid + '.fits' 

	"""
	spArc_file = '/uufs/astro.utah.edu/common/astro_data2/SDSS3/BOSS/bossredux/v5_4_14/4010/spArc-r1-00115982.fits.gz'
	spFlat_file = '/uufs/astro.utah.edu/common/astro_data2/SDSS3/BOSS/bossredux/v5_4_14/4010/spFlat-r1-00115981.fits.gz'
	image_file = '/uufs/astro.utah.edu/common/home/u0657636/final codes/sdssproc_files/imageR1ver2.fits'
	invvr_file = '/uufs/astro.utah.edu/common/home/u0657636/final codes/sdssproc_files/invrR1ver2.fits'
	"""

	# Data from sdR files 
	data = pf.open(data_file)
	biasSubImg = data[0].data

	# inverse variance
	invvr = pf.open(data_file)
	invvr = invvr[0].data

	# Data from spArc files
	[goodwaveypos, good_wavelength, wavelength, arcSigma] = dataspArc(spArc_file)

	# Data from spFlat files
	[fiberflat, xpos_final, flatSigma] = dataspFlat(spFlat_file)

	# allocating space for variables
	yvalues = n.arange(0,ypoints,1)
	GHparam = n.zeros((nwavelen,nbund,15,15))
	fib = n.arange(0,fibNo,1) 
	ngoodwave=len(good_wavelength)
	chi_sqrd = n.zeros((nwavelen,nbund,1))

	coeffAll  = n.zeros((fibBun,ypoints,15,15))
	xcenter  = n.zeros((nwavelen,nbund,fibBun))
	ycenter  = n.zeros((nwavelen,nbund,fibBun))
	sigma  = n.zeros((nwavelen,nbund,fibBun))

	# define max order
	maxorder = 4	
	
	for i_bund in range(0, 1):
	    kcons = i_bund*fibBun
	    for i_actwave in range(0, nwavelen):
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
	        fibcons = n.arange(kcons,kcons+fibBun,1)	
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
					[ff, basisfunc, basisimage,flag,xcenarr,ycenarr, sigmaarr] = create_basisfunc(actwavelength,good_wavelength,goodwaveypos,xpos_final,arcSigma,flatSigma,rowimage,x1im,x2im,y1im,y2im,fiberflat,kcons,mor,nor) 	
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
		[theta,t2,l1,t1,t2] = calparam(B1,N,p1)
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
		
		degree = 3
		
		# wavelength indexes which do not have outliers 
		reqwave = array([11,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 32, 33, 34,  41, 44, 46, 47, 48, 49, 50, 52, 53,55, 56, 57, 59, 60,62])

	# loop to get the value of PSF parameters at all wavelengths		    
	for i_fib in range(0, fibBun):
		for i_plot in range(1, len(mm)):
			wavecons   =  wavelength[:, i_bund*fibBun + i_fib]
			param = GHparam[reqwave, i_bund, mm[i_plot], nn[i_plot]]
			z = n.polyfit(good_wavelength[reqwave], param, degree)
			p = n.poly1d(z)
			coeffAll[i_bund*fibBun + i_fib, :, mm[i_plot], nn[i_plot]] = p(wavecons)
			
	
	PSFArc = createPSFArc(arcid,GHparam,xcenter, ycenter, sigma,good_wavelength,mm,nn)
	PSFBasis = createPSFBasis(arcid,coeffAll, wavelength, xpos_final, flatSigma,good_wavelength,mm,nn)
	return (PSFArc , PSFBasis)

# Function creates basis function at known wavelengths of arc-frames
def create_basisfunc(actwavelength,good_wavelength,goodwaveypos,xpos_final,arcSigma,flatSigma,rowimage,x1im,x2im,y1im,y2im,fiberflat,kcons,mor,nor):
	datacenterval = n.zeros((fibBun,1))
	fibcons = n.arange(kcons,kcons+fibBun,1) 
	n_k = n.zeros((len(fibcons),1))
	numx = n.shape(rowimage)[1]
	numy = n.shape(rowimage)[0]
	numk = len(fibcons) 
	numlambda_val = 1		
	basisfunc = n.zeros((numlambda_val,numk,numy,numx)) 
	zero1 = n.zeros((numlambda_val,numk,numy,numx)) 
	zero2 = n.zeros((numy*numx,1)) 	
	basisstack = 0
	xcenarr = n.zeros((fibBun))	
	ycenarr = n.zeros((fibBun))	
	sigmaarr = n.zeros((fibBun))	
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
                        xPSF = n.zeros((len(x),1))
                        yPSF = n.zeros((len(y),1))
			xPSF[:,0] = GH.pgh(x,cenF,sigmaF,mor)
                        yPSF[:,0] = GH.pgh(y,cenA,sigmaF,nor)
                        out = n.outer(yPSF,xPSF)
			datacenterval[i_k,0] = rowimage[starty,startx]
			basisfunc[0,i_k,starty+y[0]:(starty+y[len(y)-1]+1),startx+x[0]:(startx+x[len(x)-1]+1)] = out
			func_n_k = interpolate.interp1d(yvalues,fiberflat[fibcons[i_k],:])
			n_kVal = func_n_k(ypix)
			# relative fiber throughput (n_k) taken as 1
			n_k[i_k,0] = 1
	n_kzero= where(n_k[:,0] == 0)
	if (shape(n_kzero)[1] > 0 ):
		flag = 0
		return(n_k,zero1,zero2,flag,xcenarr,ycenarr,sigmaarr)
	else:	
		basisimage = basisimg(basisfunc, n_k)
		flag = 1		
		return (n_k,basisfunc, basisimage,flag,xcenarr, ycenarr, sigmaarr) 	

# supress the four dimensional basis function to two-dimensional
def basisimg(A00, n_k):
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

def calparam(B,N,p1):
	l1 = n.dot(n.transpose(B),N)
	t1 = n.dot(l1,p1)
	t2 = n.dot(l1,B)
	theta = n.dot(linalg.pinv(t2),t1)
	return(theta,t2,l1,t1,t2)
	
# write to FITS file
def createPSFBasis(idlstring,coeffAll, wavelength, xpos_final, flatSigma,good_wavelength,mm,nn):
	theta0 = n.zeros((fibBun,ypoints))
	theta1 = n.zeros((fibBun,ypoints))
	theta2 = n.zeros((fibBun,ypoints))
	theta3 = n.zeros((fibBun,ypoints))
	theta4 = n.zeros((fibBun,ypoints))
	theta5 = n.zeros((fibBun,ypoints))
	theta6 = n.zeros((fibBun,ypoints))
	theta7 = n.zeros((fibBun,ypoints))
	theta8 = n.zeros((fibBun,ypoints))
	theta9 = n.zeros((fibBun,ypoints))
	theta10 = n.zeros((fibBun,ypoints))
	theta11 = n.zeros((fibBun,ypoints))
	theta12 = n.zeros((fibBun,ypoints))
	theta13 = n.zeros((fibBun,ypoints))
	theta14 = n.zeros((fibBun,ypoints))
	final_wavelength  = n.zeros((fibBun,ypoints))

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
		
	a = n.arange(0,ypoints, dtype=float)
	b=n.transpose(a)
	q = b.repeat(fibBun)
	a = reshape(q, (ypoints,fibBun))
	ycenterf  = n.transpose(a)
		
	xcenterf = n.transpose(xpos_final[:,0:fibBun])
	sigmaarrf = n.transpose(flatSigma[:,0:fibBun])	
	final_wavelength= n.transpose(log10(wavelength[:,0:fibBun]))

	hdu0 = pf.PrimaryHDU(xcenterf)
	hdu0.header.update('PSFTYPE', 'GAUSS-HERMITE', 'GAUSS-HERMITE POLYNOMIALS') 
	hdu0.header.update('NPIX_X','4114', 'number of image pixels in the X-direction')
	hdu0.header.update('NPIX_Y', str(ypoints), 'number of image pixels in the Y-direction')
	hdu0.header.update('NFLUX',  str(ypoints),   'number of flux bins per spectrum [NAXIS1]')
	hdu0.header.update('NSPEC',  'fibBun',  'number of spectra [NAXIS2]')
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
	fname = 'spBasisPSF-' + arcid+'.fits'
	hdulist.writeto(fname, clobber=True)
	return (fname)

def createPSFArc(idlstring,GHparam, xcenter, ycenter, sigma,good_wavelength,mm,nn):
	xcenterf  = n.zeros((fibNo,nwavelen))
	ycenterf  = n.zeros((fibNo,nwavelen))
	sigmaarrf  = n.zeros((fibNo,nwavelen))
	theta0 = n.zeros((fibNo,nwavelen))
	theta1 = n.zeros((fibNo,nwavelen))
	theta2 = n.zeros((fibNo,nwavelen))
	theta3 = n.zeros((fibNo,nwavelen))
	theta4 = n.zeros((fibNo,nwavelen))
	theta5 = n.zeros((fibNo,nwavelen))
	theta6 = n.zeros((fibNo,nwavelen))
	theta7 = n.zeros((fibNo,nwavelen))
	theta8 = n.zeros((fibNo,nwavelen))
	theta9 = n.zeros((fibNo,nwavelen))
	theta10 = n.zeros((fibNo,nwavelen))
	theta11 = n.zeros((fibNo,nwavelen))
	theta12 = n.zeros((fibNo,nwavelen))
	theta13 = n.zeros((fibNo,nwavelen))
	theta14 = n.zeros((fibNo,nwavelen))
	final_wavelength  = n.zeros((fibNo,nwavelen))

	for i_wave in range(0, nwavelen):
		theta0[:,i_wave]  = GHparam[i_wave, : , mm[1],nn[1]].repeat(fibBun)
		theta1[:,i_wave]   = GHparam[i_wave, : , mm[2],nn[2]].repeat(fibBun)
		theta2[:,i_wave] =  GHparam[i_wave, : , mm[3],nn[3]].repeat(fibBun)
		theta3[:,i_wave] =  GHparam[i_wave, : , mm[4],nn[4]].repeat(fibBun)
		theta4[:,i_wave] =  GHparam[i_wave, : , mm[5],nn[5]].repeat(fibBun)
		theta5[:,i_wave]  =  GHparam[i_wave, : , mm[6],nn[6]].repeat(fibBun)
		theta6[:,i_wave] =  GHparam[i_wave, : , mm[7],nn[7]].repeat(fibBun)
		theta7[:,i_wave] =  GHparam[i_wave, : , mm[8],nn[8]].repeat(fibBun)
		theta8[:,i_wave] =  GHparam[i_wave, : , mm[9],nn[9]].repeat(fibBun)
		theta9[:,i_wave] =  GHparam[i_wave, : , mm[10],nn[10]].repeat(fibBun)
		theta10[:,i_wave] =  GHparam[i_wave, : , mm[11],nn[11]].repeat(fibBun)
		theta11[:,i_wave] =  GHparam[i_wave, : , mm[12],nn[12]].repeat(fibBun)
		theta12[:,i_wave] =  GHparam[i_wave, : , mm[13],nn[13]].repeat(fibBun)
		theta13[:,i_wave] =  GHparam[i_wave, : , mm[14],nn[14]].repeat(fibBun)
		theta14[:,i_wave] =  GHparam[i_wave, : , mm[15],nn[15]].repeat(fibBun)
		
		xcenterf[:,i_wave]  = n.ravel(xcenter[i_wave, :, :])
		ycenterf[:,i_wave]  = n.ravel(ycenter[i_wave, :, :])
		
		final_wavelength[:,i_wave] = (log10(good_wavelength[i_wave])).repeat(fibNo)
		
		sigmaarrf[:,i_wave]  = n.ravel(sigma[i_wave, :, :])

		
	# write to FITS file
	hdu0 = pf.PrimaryHDU(xcenterf)
	hdu0.header.update('PSFTYPE', 'GAUSS-HERMITE', 'GAUSS-HERMITE POLYNOMIALS') 
	hdu0.header.update('NFLUX',  str(nwavelen),   'number of flux bins per spectrum [NAXIS1]')
	hdu0.header.update('NSPEC',  str(fibNo),  'number of spectra [NAXIS2]')
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

	fname = 'spArcPSF-' + arcid +'.fits'
	hdulist.writeto(fname, clobber=True)
	return (fname)
	
def dataspArc(spArc_file):
	h_spArc = pf.open(spArc_file)

	# Number of fibers 
	image = h_spArc[0].data
	nfib = n.shape(image)[0]
	ndatapt = n.shape(image)[1]

	# y-sigma values
	ysigma = h_spArc[4].data
	xmin =  ysigma.field(1)  
	xmax =  ysigma.field(2)
	coeff = ysigma.field(3)
	coeffDim = n.shape(coeff)[1]
	#print coeffDim
	legendreDim = coeffDim/nfib
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

	r = n.transpose(coeff[0,:,:])
	Ysig = n.dot(r,l[1:])
	Ysig = n.transpose(Ysig)
	arcSigma =Ysig

	#wavelength value at each Y-center
	waveset 	= h_spArc[2].data
	xminwave 	= waveset.field(1)
	xmaxwave 	= waveset.field(2)
	coeffwave 	= waveset.field(3)
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
	good_lambda_val = h_spArc[1].data
	good_wavelength = good_lambda_val[:,0]
	wavenum = n.shape(good_lambda_val)[0]
	lambdadim2 = n.shape(good_lambda_val)[1]
	goodwaveypos = good_lambda_val[0:wavenum, 1:lambdadim2]
	return(goodwaveypos, good_wavelength, wavelength, arcSigma)

def dataspFlat(spFlat_file):
	h_spFlat = pf.open(spFlat_file)
	
	# Number of fibers 
	image = h_spFlat[0].data
	nfib = n.shape(image)[0]
	ndatapt = n.shape(image)[1] 
	
	# X- sigma 
	sigma	=  h_spFlat[3].data
	xminSig 	=  sigma.field(1)
	xmaxSig 	=  sigma.field(2)
	coeffXSig 	=  sigma.field(3)
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
	h_spFlat = pf.open(spFlat_file)
	peakPos = h_spFlat[1].data
	xmin =  peakPos.field(1)
	xmax =  peakPos.field(2)
	coeffxpos = peakPos.field(3)
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
	fiberflat = h_spFlat[0].data
	return(fiberflat, xpos_final, flatSigma)
	
