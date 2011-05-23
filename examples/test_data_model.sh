#!/bin/bash

#- Run this script to generate examples of the data model files
#- and exercise various steps of the code

#- Filenames
PSF_FILE="spBasisPSF-Gauss2D-TEST.fits"
SPEC_FILE="input_spectra-TEST.fits"
IMAGE_FILE="sdProc-Gauss2D-TEST.fits"
XSPEC_FILE="extracted_spectra-TEST.fits"
MODEL_IMAGE_FILE="sdProc-model-TEST.fits"

echo "-- Generating 2D Gaussian PSF"
python make_gauss2d_psf.py -o $PSF_FILE

echo "-- Generating test spectra"
python make_test_spectra.py -p $PSF_FILE -o $SPEC_FILE

echo "-- Projecting those spectra into pixels using the PSF"
python spec2pix.py -i $SPEC_FILE -p $PSF_FILE -o $IMAGE_FILE --hdu 0 --noise

echo "-- Extracting spectra from the image using the PSF"
python pix2spec.py -i $IMAGE_FILE -p $PSF_FILE -o $XSPEC_FILE

echo "-- Generating model image"
python spec2pix.py -i $XSPEC_FILE -p $PSF_FILE -o $MODEL_IMAGE_FILE --hdu 3

echo "-- Output files:"
echo "   Gauss2D PSF           : $PSF_FILE"
echo "   Input spectra         : $SPEC_FILE"
echo "   Spectra x PSF = Image : $IMAGE_FILE"
echo "   Extracted spectra     : $XSPEC_FILE"
echo "   Model image           : $MODEL_IMAGE_FILE"

echo "-- Plotting results"
python plot_test_results.py -s $SPEC_FILE -x $XSPEC_FILE -i $IMAGE_FILE \
    -p $PSF_FILE -m $MODEL_IMAGE_FILE

