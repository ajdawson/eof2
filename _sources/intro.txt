Introduction
============

:py:mod:`eof2` is a Python package for EOF analysis of spatial-temporal data. Some of the key features of :py:mod:`eof2` are:

* **Suitable for large data sets:** computationally efficient for the large output data sets of modern climate models.
* **Transparent handling of missing values:** missing values are removed automatically when computing EOFs and placed back into output fields.
* **Automatic metadata:** metadata from input fields is used to construct metadata for output fields (requires the :py:mod:`cdms2` module).
* **No Fortran dependencies:** a fast implementation written in Python using the power of :py:mod:`numpy`, so no compilers required.


Download and Installation
-------------------------

The package can be downloaded from `github <http://github.com/ajdawson/eof2>`_. To get the latest source and install the module on your system run:

.. code-block:: bash

   $ git clone git://github.com/ajdawson/eof2.git
   $ cd eof2
   $ sudo python setup.py install


Getting Started
---------------

The :py:mod:`eof2` package provides two interfaces for EOF analysis: one for :py:mod:`numpy` arrays or masked arrays; and one for :py:mod:`cdms2` variables, which preserves metadata. The two interfaces support exactly the same sets of operations.

Regardless of which interface you use, the basic usage is the same. The EOF analysis is handled by :py:class:`eof2.Eof` (or :py:class:`eof2.EofSolve` for the :py:mod:`numpy` interface). The EOF solution is computed when an instance of :py:class:`eof2.Eof` (or :py:class:`eof2.EofSolve`) is initialized. Method calls are then used to return quantities of interest.

The following is a very simple illustrative example which computes the leading 2 EOFs of a temporal spatial field:

.. code-block:: python

   import cdms2
   from eof2 import Eof

   # Read a spatial-temporal field. Time must be the first dimension.
   ncin = cdms2.open('sst_monthly.nc')
   sst = ncin('sst')
   ncin.close()

   # Initialize and Eof object. Square-root of cosine of latitude weights
   # are used.
   solver = Eof(sst, weights='coslat')

   # Retrieve the first two EOFs.
   eofs = solver.eofs(neofs=2)


Requirements
------------

This package requires as a minimum that you have `numpy <http://http://numpy.scipy.org/>`_ available. The metadata enabled interface can only be used if the :py:mod:`cdms2` module is also available. This module is distributed as part of the `UV-CDAT <http://uv-cdat.llnl.gov>`_ project. It is also distributed as part of the `cdat_lite <http://proj.badc.rl.ac.uk/cedaservices/wiki/CdatLite>`_ package.


Developing and Contributing
---------------------------

All development is done through the `github <http://github.com/ajdawson/eof2>`_ system. To check out the latest sources run:

.. code-block:: bash

   $ git clone git://github.com/ajdawson/eof2.git

Please file bug reports and feature requests using the github `issues <http://github.com/ajdawson/eof2/issues?state=open>`_.

