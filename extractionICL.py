#########################################################################################################
# READ ME:                                                                                              #
# Functions needed to make spectral extractions on MUSE IFU data cubes.                                 #
#                                                                                                       #
# Functions:                                                                                            #
## Extraction(): Make a 1D spectral extraction of MUSE IFU data on every spectral slice of a datacube.  #
#                                                                                                       #
## segfile2Ap(): 2D Gaussian convolution of a segmentation file which is converted                      #
##               to a astropy RegionMask                                                                # 
#                                                                                                       #
## twoDfits_CutOut(): Make a 2D cutout of a FITS file and save as new FITS file with updated header     #
##                    With the option to project the FITS into a new FITS frame                         #
#########################################################################################################

# Dependencies
import numpy as np
from reproject import reproject_interp

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel, convolve

import regions
from regions import Regions, RegionBoundingBox, RegionMask, PixCoord
from regions import CircleSkyRegion, RectangleSkyRegion, PolygonSkyRegion, EllipseSkyRegion
from regions import CirclePixelRegion, RectanglePixelRegion, PolygonPixelRegion, EllipsePixelRegion

from photutils import CircularAperture, SkyCircularAperture, aperture_photometry
from photutils import EllipticalAperture, SkyEllipticalAperture
from photutils import RectangularAperture, SkyRectangularAperture

def Extraction(Slicedata, var, w, regfile,
               regPath='/Users/useryou/Grad/Research/RGB/Region_files/'):
    """ 
    Output: 
        Two dictionaries, one for the IFU data and one for the error cube, keys = spectral slice
        index begins at 1, values = IFU/error data.
    ------------------------------------------
    Args:
        Slicedata (): 
        var ():
        hdr_extn ():
        regPath (str): Pointing to DS9 region files to be used as subregions/sudo-slits.
        regfile (str): DS9 region filename which defines the subregions/sudo-slits to be extracted.
    ------------------------------------------    
    Returns:
        ApSum_dict (dictionary): 1D spectra extracted from IFU data 
        Error_dict (dictionary): Error data propogated by adding in quadrature
    """
    # Separate each slice on the spectral slice
    #Slicedata = [cube[specSlice,:,:] for specSlice in range(len(wvl))]
    #var = [error_cube[specSlice,:,:] for specSlice in range(len(wvl))]
    
    # Regions read in from ds9 come out as ShapeSkyRegion(s)!
    regions = Regions.read(regPath + regfile, format='ds9')
    #w = WCS(hdr_extn[1].header) # Galfit imgBlock raw data extension = 1

    # Store the sums
    ApSum_dict = {(Ap+1):[] for Ap in range(len(regions))}
    Error_dict = {(Ap+1):[] for Ap in range(len(regions))}

    # Make the apertures & do the photometry 
    i=0
    for Region in regions:
        i+=1
        if isinstance(Region, PolygonSkyRegion):
            # Using PolygonSkyRegions 
            x,y = skycoord_to_pixel(Region.vertices, w.celestial)
            vertices = PixCoord(x,y)
            aperture = PolygonPixelRegion(vertices=vertices)
            mask = aperture.to_mask(mode='subpixels') # 'exact' not for polgon can use 'center' or 'subpixel'
            # Operations on each slice
            for idx in range(len(Slicedata)):
                # Flux
                cut = mask.cutout(Slicedata[idx])
                valuedSum = np.nansum(mask.multiply(Slicedata[idx]), axis=(0,1)) # Collapse to 1D

                # Error, added in quadrature to account for the collapse to 1D
                variance = mask.cutout(var[idx])
                Valvar = mask.multiply(var[idx])
                SumVar = np.sqrt(np.nansum(Valvar**2, axis=(0,1)))

                # Save it in the dictionary
                Apsuum = ApSum_dict[i].append(valuedSum)
                errr = Error_dict[i].append(SumVar)
        elif isinstance(Region, EllipseSkyRegion):
            SkyAp = SkyEllipticalAperture(Region.center, Region.width, 
                                          Region.height, Region.angle)
            aperture = SkyAp.to_pixel(w.celestial)
            for idx in range(len(Slicedata)):
                specSlice = aperture_photometry(Slicedata[idx], aperture, error=var[idx], wcs=w.celestial, method='exact')
                fluxSum = specSlice['aperture_sum'][0]
                varSum = specSlice['aperture_sum_err'][0]
                ApSum_dict[i].append(fluxSum)
                Error_dict[i].append(varSum)
        elif isinstance(Region, CircleSkyRegion):
            SkyAp = SkyCircularAperture(Region.center, r=Region.radius)
            aperture = SkyAp.to_pixel(w.celestial) # This does not work unless you use wcs.celestial
            for idx in range(len(Slicedata)):
                specSlice = aperture_photometry(Slicedata[idx], aperture, error=var[idx], wcs=w.celestial, method='exact')
                fluxSum = specSlice['aperture_sum'][0]
                varSum = specSlice['aperture_sum_err'][0]
                ApSum_dict[i].append(fluxSum)
                Error_dict[i].append(varSum)
        elif isinstance(Region, RectangleSkyRegion):
            SkyAp = SkyRectangularAperture(Region.center, Region.width, Region.height, Region.angle)
            aperture = SkyAp.to_pixel(w.celestial)
            for idx in range(len(Slicedata)):
                specSlice = aperture_photometry(Slicedata[idx], aperture, error=var[idx], wcs=w.celestial, method='exact')
                fluxSum = specSlice['aperture_sum'][0]
                varSum = specSlice['aperture_sum_err'][0]
                ApSum_dict[i].append(fluxSum)
                Error_dict[i].append(varSum)
        elif isinstance(Region, CirclePixelRegion):
            aperture = CircularAperture((Region.center.x, Region.center.y), Region.radius)
            for idx in range(len(Slicedata)):
                specSlice = aperture_photometry(Slicedata[idx], aperture, error=var[idx], wcs=w.celestial, method='exact')
                fluxSum = specSlice['aperture_sum'][0]
                varSum = specSlice['aperture_sum_err'][0]
                ApSum_dict[i].append(fluxSum)
                Error_dict[i].append(varSum)
    return ApSum_dict, Error_dict

def twoDfits_CutOut(fitsPath, position, size, new_filename, projectionPath=None):
    """ 
    Output: Make a 2D square cutout of a fits saved as a new fits file with updated header
            If projectionPath does not equal None: The fits from fitsPath is projected into
            the coordinate system of the fits from projectionPath
    ------------------------------------------
    Args:
        fitsPath (file path to fits): Path to the desired fits file
        position: Center position of the the fits from fitsPath
                  If projectionPath does not equal None: Center position in reprojections  
                  coordinate system e.g. in coordinates from the fits from projectionPath.
        size (int or (ny, nx)): Input fed to Cutout2D. Either int or ordered pair of ints. 
                                Size of the cutout along each axis 
                                [pixels (Default) or astropy Quantity object of angular units]
        filename (string): New name to save the cutout fits with updated header to. Will be 
                           saved with 'cutout.fits' ending.
        projectionPath (file path): File path to the fits you want to project the fits from 
                                    fitsPath into. New coordinate system or differing resolution. 
    ------------------------------------------    
    Returns:
        seg_mask (RegionMask): 1 = background, 0 = object
    """
    hdu = fits.open(fitsPath)[0] # open the segmentation fits
    wcs = WCS(hdu.header)
    
    if projectionPath != None:
        newhdu = fits.open(projectionPath)[1]
        new_wcs = WCS(newhdu.header).dropaxis(2) # Assume MUSE data cube and keep only 2 spatial dim

        # Project fits (from fitsPath) on to fits (from ProjectionPath) coordinate system
        projection, _ = reproject_interp((hdu.data, wcs),
                                            new_wcs, shape_out=newhdu.data.shape[1:])
        cut = Cutout2D(projection, position=position, size=size, wcs=new_wcs) # make the cutout
    else:
        cut = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)

    # Save cutout to fits with updated wcs info
    cut.data[(cut.data > 0)] = 1.
    hdu.data = cut.data
    hdu.header.update(cut.wcs.to_header())
    filename = new_filename + '_cutout.fits'
    hdu.writeto(filename, overwrite=True)
    new = fits.open(filename)
    return new

def Segfile2Ap(seg_data, stddev, idx=None):    
    """ 
    Output: Convert SExtractor segmentation file to an astropy region mask and convlve 
            the seg with a 2D Gaussian kernel of size stddev
    ------------------------------------------
    Args:
        seg_data(np.ndarray): data from the SExtractor segmentation fits file.
        stddev (int/float): # of standard deviations to create the 2D Gaussian kernel.
          see: https://docs.astropy.org/en/stable/api/astropy.convolution.Gaussian2DKernel.html#
        idx (tuple of ordered pairs (int)): Indices used to slice the ndarray (seg_data) 
          ex: idx = ((y1, y2), (x1, x2)) --> seg_data[y1:y2, x1:x2]
          Default: None --> whole ndarry is the desired mask
    ------------------------------------------    
    Returns:
        seg_mask (RegionMask): 1 = background, 0 = object
    """
    # Create the 2D Gaussian kernel
    kernelGaussian = Gaussian2DKernel(x_stddev=stddev, y_stddev=stddev)
    seg_conv = convolve(seg_data, kernelGaussian)
    # background values are between 0 to 0.1, sp 0 +/- 0.05
    bck_val = np.isclose(seg_conv, 0., atol=0.05)
    #swap 0's for 1's to vlaue the background in the region mask
    conv_zero = np.where(bck_val, 1., 0.)
    conv_zero[conv_zero < 1.] = np.nan # make 0's nans to unbias the mean
    # Set the correct index/pixel of the image to the bounding box for the RegionMask
    if idx != None:
        y1 = idx[0][0] # bounds of the ndarray to be used as the mask 
        y2 = idx[0][1] 
        x1 = idx[1][0]
        x2 = idx[1][1]
        bbox = RegionBoundingBox(ixmin=x1, ixmax=x2, 
                             iymin=y1, iymax=y2)
    else:
        bbox = RegionBoundingBox(ixmin=0, ixmax=conv_zero.shape[1], 
                                 iymin=0, iymax=conv_zero.shape[0])
    
    # Make the array a region mask
    seg_mask = RegionMask(conv_zero, bbox)
    return seg_mask, seg_conv

def bin_data(spec, wavelengths, err, n, var = True):
    """ 
    Output: Data variance weighted and summed in bins of n
    ------------------------------------------
    Args:
        spec: The spectra data
        wavelengths: The wavelength range of the spectra
        err: The variance associated with each point in spec
        ## len(spec) = len(wavelengths) = len(err)
        n: The number of entries that will be summed into one bin
        var: if var = False the err is a std and will be converted to a variance
             If var = True the err is a variance and won't be changed (Default)
    ------------------------------------------    
    Returns:
        np.array of :
         * spectra data (Inverse variance weighted) 
         * wavelengths (Binned and averaged by bin width)
         * variance (Binned and averaged by bin width)
         len(spec) == len(wave) == len(var)
    """
    err = np.float128(err)
    if var == True:
        var = err
    else:
        var = err**2
    inv_var = 1/var
    # y/sigma**2
    weighted_pts = (spec*inv_var)
    # What is the bin width in AA? wvl_increment * binning_increment
    bin_increment = [round((wavelengths[i+1]-wavelengths[i]).value,3) for i in range(len(wavelengths[:-1]))]
    binInc_unique = np.unique(bin_increment)
    if len(binInc_unique) > 1:
        print('Wavelength increment is not unique:', binInc_unique)
    else:
        bin_width = binInc_unique*n # AA
    # Sum every nth entry
    # Take average to make lengths match of wave and var
    binned_wave = np.add.reduceat(wavelengths, np.arange(0, len(wavelengths), n))/n 
    # Prop. error 1/(sum (1/sigma**2))
    binned_var = (np.add.reduceat(1/var.value, np.arange(0, len(var), n)))**(-1)*(u.erg/(u.s*u.cm**2*u.AA))**2
    # (1/var**2) summed every nth entry then inverted
    den = np.add.reduceat(inv_var, np.arange(0, len(inv_var), n))
    # Inverse variance weighting
    binned_data = np.add.reduceat(weighted_pts, np.arange(0, len(weighted_pts), n))/den
    # If n = odd and len(wavelength) = even --> One wave binned alone
    # Throw the odd man out, binning func takes average
    if np.isclose(wavelengths[-1].value, binned_wave[-1].value, atol = 100) == False:
        binned_data = binned_data[:-1]
        binned_wave = binned_wave[:-1]
        binned_var = binned_var[:-1]
    # Sanity checks
    if binned_data.shape != binned_wave.shape != binned_var.shape:
        raise ValueError("Dimensions are mismatched. Got data.dim:{}, wave.dim:{}, var.dim:{}"
                         .format(binned_data.shape, binned_wave.shape, binned_var.shape))
    if binned_wave.unit != u.AA:
        raise ValueError("Wrong wavelength units")
    if binned_var.unit != (u.erg/(u.s*u.cm**2*u.AA))**2:
        raise ValueError("Wrong variance units:{}. Should be {}."
                         .format(binned_var.unit, (u.erg/(u.s*u.cm**2*u.AA))**2))
    return binned_data, binned_wave, binned_var, bin_width

def convertNbin_Bckground(spec, wavelength, var, n):
    spectra, wave, variance, bin_width = bin_data(np.array(spec)*1e-20*u.erg/(u.s*u.cm**2*u.AA), 
                                                  wavelength, 
                                                  np.array(var)*1e-40*(u.erg/(u.s*u.cm**2*u.AA))**2, n, var=True)
    return spectra, wave, variance, bin_width
