import lightkurve
import numpy as np
from scipy.linalg import lstsq

class bkgd_tpf(lightkurve.TessTargetPixelFile):

	def get_bkgd(self):
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
		rawflux = flux + oldbkgd

		# Step 1: Define new background pixels (or take mask that is passed in)

		# Find background flux level
		sorted_median_pixel_values = np.ravel(np.nanmedian(rawflux, axis=0))[(-np.nanmedian(rawflux,axis=0)).argsort(axis=None)]
		hist,bins = np.histogram(sorted_median_pixel_values, bins = np.arange(0,200,1))
		median_bkgd_lvl = np.argmax(hist)

		# Create background pixel mask
		bkgdmask = np.ones((flux.shape[1],flux.shape[2]),dtype='bool')

		bkgdmask[np.isnan(np.nanmedian(rawflux,axis=0))] = False
		bkgdmask[np.nanmedian(rawflux,axis=0) > median_bkgd_lvl+5] = False
		bkgdmask[np.nanmedian(rawflux,axis=0) < median_bkgd_lvl-5] = False

		# Fit to backgrouond pixel values
		zz = np.nanmedian((rawflux[:,bkgdmask]),axis=0)
		z = (rawflux[:,bkgdmask])

		z=z[:,np.isfinite(zz)].T
		x,y = np.meshgrid(np.arange(flux.shape[1]), np.arange(flux.shape[2]))
		x=x.T[newmask].flatten()[np.isfinite(zz)]
		y=y.T[newmask].flatten()[np.isfinite(zz)]

		ind = np.isfinite(np.sum(z,axis=0))

		M = np.c_[x*y,x,y,np.ones(x.shape[0])]
		p = np.zeros((4,flux.shape[0]))
		p[:,ind], _, _, _ = lstsq(M, z[:,ind])

		x = np.tile(np.expand_dims(np.tile(np.expand_dims(np.arange(flux.shape[1]),axis=1),flux.shape[0]).T,axis=2),flux.shape[2])
		y = np.tile(np.expand_dims(np.tile(np.expand_dims(np.arange(flux.shape[2]),axis=1),flux.shape[1]),axis=2),flux.shape[0]).T
		c = np.tile(np.expand_dims(np.tile(np.expand_dims(p,axis=2),flux.shape[1]),axis=3),flux.shape[2])

		bkgd = c[0,:,:,:]*x*y + c[1,:,:,:]*x + c[2,:,:,:]*y + c[3,:,:,:]

		return bkgd
