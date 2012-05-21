import os
import sys
import urllib2

import cdms2
import cdutil
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from eof2 import Eof


if __name__ == "__main__":

    # Download SLP data.
    remote_file = os.path.join("ftp://ftp.cdc.noaa.gov", "Datasets",
               "ncep.reanalysis.derived", "surface", "slp.mon.mean.nc")
    local_file = "slp.mon.mean.nc"
    if not os.path.exists(local_file):
        ncremote = urllib2.urlopen(remote_file)
        nclocal = open(local_file, "wb")
        nclocal.write(ncremote.read())
        nclocal.close()
        ncremote.close()

    # Read SLP data.
    ncin = cdms2.open(local_file, "r")
    slp = ncin("slp")
    ncin.close()

    # Extract DJF seasons.
    cdutil.setTimeBoundsMonthly(slp)
    slp_djf = cdutil.times.DJF(slp)
    slp_djf_mean = cdutil.averager(slp_djf, axis="t")
    slp_djf = slp_djf - slp_djf_mean
    slp_djf.id = "slp"

    # Do EOF analysis on European/Atlantic sector.
    slp_djf_eat = slp_djf(latitude=(30,90,"ccb"), longitude=(-80,40,"ccb"))
    solver = Eof(slp_djf_eat, weights="area")
    eof1 = solver.eofsAsCovariance(neofs=1)

    # Plot the leading EOF
    m = Basemap(projection="ortho", lat_0=60., lon_0=-20.)
    lons, lats = eof1.getLongitude()[:], eof1.getLatitude()[:]
    x, y = m(*np.meshgrid(lons, lats))
    m.contourf(x, y, eof1(squeeze=True), cmap=plt.cm.RdBu_r)
    m.drawcoastlines()
    m.drawparallels(np.arange(-80, 90, 20))
    m.drawmeridians(np.arange(0, 360, 20))
    plt.title("EOF1 expressed as covariance", fontsize=16)
    plt.savefig("example0_0.png")

