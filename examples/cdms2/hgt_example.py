"""
Compute and plot the leading EOF of geopotential height on the 500 hPa
pressure surface over the European/Atlantic sector during winter time.

This example uses the metadata-retaining cdms2 interface.

"""
import cdms2
import cdutil
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from eof2 import Eof


# Read geopotential height data using the cdms2 module from CDAT. The file
# contains December-February averages of geopotential height at 500 hPa for
# the European/Atlantic domain (80W-40E, 20-90N).
ncin = cdms2.open('../../example_data/hgt_djf.nc', 'r')
z_djf = ncin('z')
ncin.close()

# Compute anomalies by removing the time-mean.
z_djf_mean = cdutil.averager(z_djf, axis='t')
z_djf = z_djf - z_djf_mean
z_djf.id = 'z'

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
solver = Eof(z_djf, weights='coslat')

# Retrieve the leading EOF, expressed as the covariance between the leading PC
# time series and the input SLP anomalies at each grid point.
eof1 = solver.eofsAsCovariance(neofs=1)

# Plot the leading EOF expressed as covariance in the European/Atlantic domain.
m = Basemap(projection='ortho', lat_0=60., lon_0=-20.)
lons, lats = eof1.getLongitude()[:], eof1.getLatitude()[:]
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, eof1(squeeze=True), cmap=plt.cm.RdBu_r)
m.drawcoastlines()
m.drawparallels(np.arange(-80, 90, 20))
m.drawmeridians(np.arange(0, 360, 20))
plt.title('EOF1 expressed as covariance', fontsize=16)

plt.show()
