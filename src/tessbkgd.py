import lightkurve
import numpy as np
from scipy.linalg import lstsq
from astropy.units import Quantity

class bkgd_tpf(lightkurve.TessTargetPixelFile):

    def get_bkgd(self,bkgdmask = None, order='linear'):
        """Calculates a new background for a target pixel file
            Parameters
            ----------

            Returns
            -------
            bkgd : numpy array
                Array containing new background for each cadence
        """

        #Set up some arrays
        flux = np.copy(self.flux)
        oldbkgd = np.copy(self.flux_bkg)
        if np.nansum(oldbkgd) == 0:
            rawflux = flux
        else:
            rawflux = flux + oldbkgd

        if bkgdmask is None:
            # Step 1: Define new background pixels (or take mask that is passed in)
            # Find background flux level
            sorted_median_pixel_values = np.ravel(np.nanmedian(rawflux.value, axis=0))[(-np.nanmedian(rawflux.value,axis=0)).argsort(axis=None)]
            hist,bins = np.histogram(sorted_median_pixel_values, bins = np.arange(0,500,10))
            median_bkgd_lvl = bins[np.argmax(hist)]

            # Create background pixel mask
            bkgdmask = np.ones((flux.shape[1],flux.shape[2]),dtype='bool')

            bkgdmask[np.isnan(np.nanmedian(rawflux.value,axis=0))] = False
            bkgdmask[np.nanmedian(rawflux.value,axis=0) > 1.1*median_bkgd_lvl] = False
            bkgdmask[np.nanmedian(rawflux.value,axis=0) < 0.9*median_bkgd_lvl] = False


        # Fit to backgrouond pixel values
        zz = np.nanmedian((rawflux[:,bkgdmask]),axis=0)
        z = (rawflux[:,bkgdmask])

        z=z[:,np.isfinite(zz)].T
        x,y = np.meshgrid(np.arange(flux.shape[1]), np.arange(flux.shape[2]))
        x=x.T[bkgdmask].flatten()[np.isfinite(zz)]
        y=y.T[bkgdmask].flatten()[np.isfinite(zz)]

        ind = np.isfinite(np.sum(z,axis=0))

        if order == 'quadratic':
            M = np.c_[x**2,y**2,x*y,x,y,np.ones(x.shape[0])]
            p = np.zeros((6,flux.shape[0]))

            p[:,ind], _, _, _ = lstsq(M, z[:,ind])

            x = np.tile(np.expand_dims(np.tile(np.expand_dims(np.arange(flux.shape[1]),axis=1),flux.shape[0]).T,axis=2),flux.shape[2])
            y = np.tile(np.expand_dims(np.tile(np.expand_dims(np.arange(flux.shape[2]),axis=1),flux.shape[1]),axis=2),flux.shape[0]).T
            c = np.tile(np.expand_dims(np.tile(np.expand_dims(p,axis=2),flux.shape[1]),axis=3),flux.shape[2])

            bkgd = c[0,:,:,:]*x**2 + c[1,:,:,:]*y**2 + c[2,:,:,:]*x*y + c[3,:,:,:]*x + c[4,:,:,:]*y + c[5,:,:,:]
        elif order == 'linear':
            # M = np.c_[x*y,x,y,np.ones(x.shape[0])]
            # p = np.zeros((4,flux.shape[0]))
            M = np.c_[x,y,np.ones(x.shape[0])]
            p = np.zeros((3,flux.shape[0]))

            p[:,ind], _, _, _ = lstsq(M, z[:,ind])

            x = np.tile(np.expand_dims(np.tile(np.expand_dims(np.arange(flux.shape[1]),axis=1),flux.shape[0]).T,axis=2),flux.shape[2])
            y = np.tile(np.expand_dims(np.tile(np.expand_dims(np.arange(flux.shape[2]),axis=1),flux.shape[1]),axis=2),flux.shape[0]).T
            c = np.tile(np.expand_dims(np.tile(np.expand_dims(p,axis=2),flux.shape[1]),axis=3),flux.shape[2])

        # bkgd = c[0,:,:,:]*x*y + c[1,:,:,:]*x + c[2,:,:,:]*y + c[3,:,:,:]
            bkgd = c[0,:,:,:]*x + c[1,:,:,:]*y + c[2,:,:,:]
        else:
            print('Unrecognised order . Only linear and quadratic fits to the background are currently implemented.')

        if self.get_header(1).get("TUNIT5") == "e-/s":
            unit = "electron/s"
        bkgd = Quantity(bkgd, unit="electron/s")

        return bkgd,bkgdmask

    @property
    def flux(self) -> Quantity:
        """Returns the flux for all good-quality cadences."""
        unit = None
        if self.get_header(1).get("TUNIT5") == "e-/s":
            unit = "electron/s"
        return Quantity(self.hdu[1].data["FLUX"][self.quality_mask], unit=unit)

    @flux.setter
    def flux(self,value):
        self.hdu[1].data['FLUX'][self.quality_mask] = value

    @property
    def flux_bkg(self) -> Quantity:
        """Returns the background flux for all good-quality cadences."""
        return Quantity(self.hdu[1].data["FLUX_BKG"][self.quality_mask], unit="electron/s")

    @flux_bkg.setter
    def flux_bkg(self,value):
        self.hdu[1].data['FLUX_BKG'][self.quality_mask] = value

    @property
    def raw_cnts(self):
        """Returns the raw counts for all good-quality cadences."""
        return self.hdu[1].data['RAW_CNTS'][self.quality_mask]

