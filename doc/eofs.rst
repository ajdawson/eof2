EOF Analysis
============

The technique of Empirical Orthogonal Function analysis, usually just referred to as EOF analysis, is very common in geophysical sciences.
The general aim of EOF analysis is to simplify a spatial-temporal data set by transforming it to spatial patterns of variability and temporal projections of these patterns.
The spatial patterns are the EOFs, and can be thought of as basis functions in terms of variance.
The associated temporal projections are the pricipal components (PCs) and are the temporal coefficients of the EOF patterns.

EOF analysis is really just re-expressing the original data set in terms of a variance basis.
The original data set can be completely reconstructed using the EOFs and PCs.
iHowever, in practice it is often only a subset of the EOFs that are of interest.
Individual EOFs can sometimes have a physical interpretation assigned to them, or sometimes one wishes to produce a data set that is truncated in terms of variance by reconstructing the original data set using only a limited number of EOFs.


Method of Solution in :py:mod:`eof2`
------------------------------------

A method based on `singular value decomposition (SVD) <http://en.wikipedia.org/wiki/Singular_value_decomposition>`_ is used in :py:mod:`eof2` [#]_. This avoids having to compute the covariance matrix directly and is therefore optimal for data sets with a large spatial dimension.

The input to EOF analysis is a spatial-temporal field. This is represented in Python by an array (or :py:mod:`cdms2` variable) of two or more dimensions. When one inputs an array or variable into an :py:mod:`eof2` solver it is re-shaped and stored internally as a two-dimensional array where the first dimension is time and the second dimension is space. It is a formal requirement of EOF analysis that this array have columns with zero-mean. The :py:mod:`eof2` solvers will automatically attempt to subtract the mean from each column unless the keyword argument *center* is set to *False*. This may be useful when anomalies have already been computed.

Any missing values in the two-dimensional data array are identified and removed [#]_. The SVD of the anomaly matrix is then computed. The SVD computed is the truncated form, where only singular vectors (EOFs/PCs) that correspond to non-zero singular values are returned. Since the singular value is related to the fraction of variance represented by an EOF, neglecting those with singular values of zero retains a full solution. This method further reduces the computational cost of the analysis [#]_. The EOFs are the right singular vectors and the standardized PCs are the left singular vectors while the squared singular values are the variances associated with each EOF.


Mathematical Description
------------------------

Consider a data set that consists of observations of a single geophysical variable at multiple positions in space :math:`x_1, x_2, \ldots, x_M` and at multiple times :math:`t_1, t_2, \ldots, t_N,`.
These observations are arranged in a matrix :math:`\mathbf{F}` with dimension :math:`N \times M` such that each row of :math:`\mathbf{F}` is a map of observations at all points in space at a particular time, and each column is a time-series of observations at a particular point at all times.
The time-mean is then removed from of the :math:`M` time series to form the anomaly matrix :math:`\mathbf{A}` whose columns have zero-mean:

.. math::

   \mathbf{A} = \begin{pmatrix}
       a_{1,1} & a_{1,2} & \cdots & a_{1,M} \\
       a_{2,1} & a_{2,2} & \cdots & a_{2,M} \\
       \vdots  & \vdots  & \ddots & \vdots \\
       a_{N,1} & a_{N,2} & \cdots & a_{N,M}
   \end{pmatrix}

Typically one would then compute the covariance matrix :math:`\mathbf{R} = \mathbf{A}^\intercal \mathbf{A}` and solve the eigenvalue problem:

.. math::
   :label: eig

   \mathbf{R C} = \mathbf{C} \Lambda

where the columns of :math:`\mathbf{C}` are the eigenvectors (EOFs) and the eigenvalues (EOF variances) are on the leading diagonal of :math:`\Lambda`. The PCs :math:`\mathbf{P}` can then be computed from the projection of :math:`\mathbf{A}` onto the EOFs:

.. math::
   
   \mathbf{P} = \mathbf{A C}

The SVD method used by the :py:mod:`eof2` solvers is slightly different to this. Instead of computing the covariance matrix directly it computes the SVD of :math:`\mathbf{A}`:

.. math::

   \mathrm{SVD}\left(\mathbf{A}\right) = \mathbf{U} \Gamma \mathbf{V}^\intercal

The columns of :math:`\mathbf{U}` and :math:`\mathbf{V}` are the singular vectors and the singular values are on the leading diagonal of :math:`\Gamma`. To show relation between the SVD and :eq:`eig` we express the covariance matrix in two ways, first a rearranged form of :eq:`eig` [#]_:

.. math::
   :label: R1

   \mathbf{R} = \mathbf{C} \Lambda \mathbf{C}^\intercal

and second the expression for the covariance matrix :math:`\mathbf{R}` after first taking the SVD of :math:`\mathbf{A}`:

.. math::
   :label: R2

   \mathbf{R} = \mathbf{A}^\intercal \mathbf{A} = \left( \mathbf{U} \Gamma \mathbf{V}^\intercal \right)^\intercal \left( \mathbf{U} \Gamma \mathbf{V}^\intercal \right) = \mathbf{V} \Gamma^\intercal \mathbf{U}^\intercal \mathbf{U} \Gamma \mathbf{V}^\intercal = \mathbf{V} \Gamma^\intercal \Gamma \mathbf{V}^\intercal

It should be clear from :eq:`R1` and :eq:`R2` that :math:`\mathbf{C} = \mathbf{V}` and :math:`\Lambda = \Gamma^\intercal \Gamma`. An extra benefit of the SVD method is that the singular vectors in :math:`\mathbf{U}` are the standardized PCs. This can be shown by first forming an expression for :math:`\mathbf{A}` in terms of the EOFs and the PCs, note that this is the reconstruction expression:

.. math::

   \mathbf{A} = \mathbf{P} \mathbf{C}^\intercal

By defining a normalized PC :math:`\phi_j` as

.. math::

   \phi_j = \dfrac{\mathbf{p}_j}{\sqrt{\lambda_j}}

where :math:`\mathbf{p}_j` is a column of :math:`\mathbf{P}`, and defining a diagonal matrix :math:`\mathbf{D}` with :math:`\sqrt{\lambda}_j` on the leading diagonal and a matrix :math:`\Phi` of the ordered :math:`\phi_j` as columns we can write a new expression for :math:`\mathbf{A}`:

.. math::

   \mathbf{A} = \Phi \mathbf{D} \mathbf{C}^\intercal

It should be clear that this expression is equivalent to the SVD of :math:`\mathbf{A}` and therefore that the left singular vectors are the PCs scaled to unit variance.


.. rubric:: Footnotes

.. [#] Note that this is the linear algebra technique and is different to the coupled SVD analysis of two fields. The latter technique is often (and less ambiguously) referred to as maximum covariance analysis (MCA).

.. [#] An often used alternative and equally valid strategy is to set missing values to a constant value. This method yields the same solution since a constant time-series has zero-variance, but increases the computational cost of computing the EOFs.

.. [#] For an :math:`N` by :math:`M` anomaly matrix the rank of the corresponding covariance matrix can be at most :math:`\mathrm{min}\left(m,n\right)`. The number of zero eigenvalues is at least :math:`\left|m - n\right|`.

.. [#] This rearrangement is possible since the column eigenvectors in :math:`\mathbf{C}` are mutually orthogonal and hence :math:`\mathbf{C} \mathbf{C}^\intercal = \mathbf{I}` where :math:`\mathbf{I}` is the identity matrix.

