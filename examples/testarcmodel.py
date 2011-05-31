import numpy as n
import pyfits as pf

paramfile  = 'spBasisPSF-r1-00122444.fits'

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
