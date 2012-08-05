API References
==============

Reference documentation for both :py:mod:`eof2` interfaces is available:

* :ref:`cdms2-api-ref`

* :ref:`numpy-api-ref`

* :ref:`cdms2-tools-api-ref`

* :ref:`numpy-tools-api-ref`

.. note::

   The solver classes :py:class`eof2.Eof` and :py:class:`eof2.EofSolver` are
   made available at the top level of :py:mod:`eof2`, but are actually defined
   in separate modules :py:mod:`eof2.eofwrap` and :py:mod:`eof2.eofsolve`
   respectively.


.. _cdms2-api-ref:

Metadata-enabled interface
--------------------------

.. automodule:: eof2.eofwrap
   :members: Eof


.. _numpy-api-ref:

Non-metadata interface
----------------------

.. automodule:: eof2.eofsolve
   :members: EofSolver


.. _cdms2-tools-api-ref:

Extra tools (metadata-enabled)
------------------------------

.. automodule:: eof2.tools
   :members:


.. _numpy-tools-api-ref:

Extra tools (non-metadata)
--------------------------

.. automodule:: eof2.nptools
   :members:
