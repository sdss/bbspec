#!/bin/bash

#- Run this script to generate examples of the data model files
#- and exercise various steps of the code

PSF_FILE="psf_gauss2d.fits"
SPEC_FILE="test_spectra.fits"
IMAGE_FILE="test_image.fits"
XSPEC_FILE="extracted_spectra.fits"

echo "-- Generating 2D Gaussian PSF"
python make_gauss2d_psf.py -o $PSF_FILE

echo "-- Generating test spectra"
python make_test_spectra.py -p $PSF_FILE -o $SPEC_FILE

echo "-- Projecting those spectra into pixels using the PSF"
python spec2pix.py -i $SPEC_FILE -p $PSF_FILE -o $IMAGE_FILE

echo "-- Extracting spectra from the image using the PSF"
python pix2spec.py -i $IMAGE_FILE -p $PSF_FILE -o $XSPEC_FILE

echo "-- Output files:"
echo "   Gauss2D PSF           : $PSF_FILE"
echo "   Input spectra         : $SPEC_FILE"
echo "   Spectra x PSF = Image : $IMAGE_FILE"
echo "   Extracted spectra     : $XSPEC_FILE"

echo "-- Plotting results"
python plot_test_results.py -s test_spectra.fits -x extracted_spectra.fits -i test_image.fits -p psf_gauss2d.fits 

