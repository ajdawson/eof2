import os
import urllib2

import cdms2
import cdutil
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from eof2 import Eof


if __name__ == "__main__":

    # Download SST data.
    remote_file = "http://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/kaplan_sst/sst.mon.anom.nc"
    local_file = "sst.mon.mean.nc"
    if not os.path.exists(local_file):
        ncremote = urllib2.urlopen(remote_file)
        nclocal = open(local_file, "wb")
        nclocal.write(ncremote.read())
        nclocal.close()
        ncremote.close()

    # Read SST data.
    ncin = cdms2.open(local_file, "r")
    sst = ncin("sst")
    ncin.close()

    # Extract DJF seasons.
    cdutil.setTimeBoundsMonthly(sst)
    NDJFM = cdutil.times.Seasons("NDJFM")
    sst_djf = NDJFM(sst)

    # Do EOF analysis on tropical Pacific.
    sst_djf_cp = sst_djf(latitude=(-20,60,"ccb"), longitude=(120,300,"ccb"))
    solver = Eof(sst_djf_cp, weights="coslat", center=True)
    eof1 = solver.eofsAsCorrelation(neofs=1)

    # Plot the leading EOF.
    m = Basemap(projection="cyl", llcrnrlon=120, llcrnrlat=-20,
            urcrnrlon=300, urcrnrlat=60)
    lons, lats = eof1.getLongitude()[:], eof1.getLatitude()[:]
    x, y = m(*np.meshgrid(lons, lats))
    clevs = np.arange(-1, 1.1, .1)
    m.contourf(x, y, eof1(squeeze=True), clevs, cmap=plt.cm.RdBu)
    m.drawcoastlines()
    m.drawparallels(np.arange(-30,31,10))
    m.drawmeridians(np.arange(120,310,20))
    cb = plt.colorbar(orientation="horizontal")
    cb.set_label("correlation coefficient", fontsize=12)
    plt.title("EOF1 (correlation of PC1 with SST)", fontsize=16)
    plt.savefig("example1_0.png")

