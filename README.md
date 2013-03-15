eof2 - EOF analysis in Python
=============================

# Deprecation notice:

eof2 has been replaced by [eofs](https://github.com/ajdawson/eofs) and may not be maintained further. All of the features of eof2 are available in eofs.


Overview
--------

eof2 is a Python package for performing EOF analysis on spatial-temporal data sets,
licensed under the GNU GPLv3.

The package was created to simplify the process of EOF analysis in the Python
environment. Some of the key features are listed below:

* Suitable for large data sets: computationally efficient for the large data sets
  typical of modern climate model output.
* Transparent handling of missing values: missing values are removed automatically
  when computing EOFs and re-inserted into output fields.
* Automatic meta-data: if the cdms2 module (from CDAT) is available, meta-data from
  input fields is used to construct output meta-data.
* No Fortran dependencies: written in Python using the power of NumPy, no compilers
  required.

The package is designed to work both within a CDAT environment or as a stand-alone
package.


Requirements
------------

eof2 only requires the NumPy package. However, for full functionality (meta-data
interfaces) the cdms2 module is required. cdms2 is part of the Climate Data Analysis
Tools ([CDAT](http://www2-pcmdi.llnl.gov/cdat)) or can be obtained separately in the
[cdat_lite](http://proj.badc.rl.ac.uk/ndg/wiki/CdatLite) package.


Documentation
-------------

Documentation is available [online](http://ajdawson.github.com/eof2). The package
docstrings are also very complete and can be used as a source of reference when working
interactively.


Frequently asked questions
--------------------------

* **Why is it called eof2?**
  The package was originally written to be used in the CDAT environment, which already
  has a package named eof. The eof package in CDAT and eof2 are very different, so eof2
  should not be seen as the successor to that package, but rather as an independent package.
* **Do I need CDAT/cdms2 to use eof2?**
  No. All the computation code uses NumPy only. The cdms2 module is only required for the
  meta-data preserving interfaces.


Installation
------------

    sudo python setup.py install

to install system-wide, or to install in a specified location:

    python setup.py install --install-lib=/PATH/TO/INSTALL/DIR


Thanks
------

The very first version of this code was based on the SVDEOFs code from the PyClimate
project (http://www.pyclimate.org/). A big thanks to those guys for contributing their
code to the community.

