import numpy as n
import pyfits as pf

paramfile  = 'demo1.fits'

data = pf.open(paramfile)
theta1 = data[5].data
theta2 =  data[6].data
theta3 = data[7].data
theta4 = data[8].data
theta5 =data[9].data
theta6 =data[10].data
theta7 = data[11].data
theta8 = data[12].data
theta9 = data[13].data
theta10 = data[14].data
theta11 = data[15].data
theta12 = data[16].data
theta13 =data[17].data
theta14 = data[18].data
"""
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
"""
