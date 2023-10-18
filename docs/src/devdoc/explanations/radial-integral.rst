.. _radial-integral:

SOAP and LODE radial integrals
===================================

On this page, we describe the exact mathematical expression that are implemented in the
radial integral and the splined radial integral classes i.e.
:ref:`python-splined-radial-integral`. Note that this page assumes knowledge of
spherical expansion & friends and currently serves as a reference page for
the developers to support the implementation.

Preliminaries
-------------

In this subsection, we briefly provide all the preliminary knowledge that is needed to
understand what the radial integral class is doing. The actual explanation for what is
computed in the radial integral class can be found in the next subsection (1.2). The
spherical expansion coefficients :math:`\langle anlm | \rho_i \rangle` are completely
determined by specifying two ingredients:

-  the atomic density function :math:`g(r)` as implemented in
   :ref:`python-atomic-density`, often chosen to be a Gaussian or Delta function, that
   defined the type of density under consideration. For a given center atom :math:`i` in
   the structure, the total density function :math:`\rho_i(\boldsymbol{r})` around is
   then defined as :math:`\rho_i(\boldsymbol{r}) = \sum_{j} g(\boldsymbol{r} -
   \boldsymbol{r}_{ij})`.

-  the radial basis functions :math:`R_{nl}(r)` as implementated
   :ref:`python-radial-basis`, on which the density :math:`\rho_i` is projected. To be
   more precise, the actual basis functions are of the form

   .. math::

      B_{nlm}(\boldsymbol{r}) = R_{nl}(r)Y_{lm}(\hat{r}),

   where :math:`Y_{lm}(\hat{r})` are the real spherical harmonics evaluated at the point
   :math:`\hat{r}`, i.e. at the spherical angles :math:`(\theta, \phi)` that determine
   the orientation of the unit vector :math:`\hat{r} = \boldsymbol{r}/r`.

The spherical expansion coefficient :math:`\langle nlm | \rho_i \rangle` (we ommit the
chemical species index :math:`a` for simplicity) is then defined as

.. math::

   \begin{aligned}
      \langle nlm | \rho_i \rangle & = \int \mathrm{d}^3\boldsymbol{r}
      B_{nlm}(\boldsymbol{r}) \rho_i(\boldsymbol{r}) \\ \label{expansion_coeff_def} & =
      \int \mathrm{d}^3\boldsymbol{r} R_{nl}(r)Y_{lm}(\hat{r})\rho_i(\boldsymbol{r}).
   \end{aligned}

In practice, the atom centered density :math:`\rho_i` is a superposition of the neighbor
contributions, namely :math:`\rho_i(\boldsymbol{r}) = \sum_{j} g(\boldsymbol{r} -
\boldsymbol{r}_{ij})`. Due to linearity of integration, evaluating the integral can then
be simplified to

.. math::

   \begin{aligned}
      \langle nlm | \rho_i \rangle & = \int \mathrm{d}^3\boldsymbol{r}
      R_{nl}(r)Y_{lm}(\hat{r})\rho_i(\boldsymbol{r}) \\ & = \int
      \mathrm{d}^3\boldsymbol{r} R_{nl}(r)Y_{lm}(\hat{r})\left( \sum_{j}
      g(\boldsymbol{r} - \boldsymbol{r}_{ij})\right) \\ & = \sum_{j} \int
      \mathrm{d}^3\boldsymbol{r} R_{nl}(r)Y_{lm}(\hat{r}) g(\boldsymbol{r} -
      \boldsymbol{r}_{ij}) \\ & = \sum_j \langle nlm | g;\boldsymbol{r}_{ij} \rangle.
   \end{aligned}

Thus, instead of having to compute integrals for arbitrary densities :math:`\rho_i`, we
have reduced our problem to the evaluation of integrals of the form

.. math::

   \begin{aligned}
      \langle nlm | g;\boldsymbol{r}_{ij} \rangle & = \int \mathrm{d}^3\boldsymbol{r}
      R_{nl}(r)Y_{lm}(\hat{r})g(\boldsymbol{r} - \boldsymbol{r}_{ij}),
   \end{aligned}

which are completely specified by

-  the density function :math:`g(\boldsymbol{r})`

-  the radial basis :math:`R_{nl}(r)`

-  the position of the neighbor atom :math:`\boldsymbol{r}_{ij}` relative to the center
   atom

The radial integral class
-------------------------

In the previous subsection, we have explained how the computation of the spherical
expansion coefficients can be reduced to integrals of the form

.. math::

   \begin{aligned}
      \langle nlm | g;\boldsymbol{r}_{ij} \rangle & = \int \mathrm{d}^3\boldsymbol{r}
      R_{nl}(r)Y_{lm}(\hat{r})g(\boldsymbol{r} - \boldsymbol{r}_{ij}).
   \end{aligned}

If the atomic density is spherically symmetric, i.e. if :math:`g(\boldsymbol{r}) = g(r)`
this integral can always be written in the following form:

.. math::

   \begin{aligned} \label{expansion_coeff_spherical_symmetric}
      \langle nlm | g;\boldsymbol{r}_{ij} \rangle & =
      Y_{lm}(\hat{r}_{ij})I_{nl}(r_{ij}).
   \end{aligned}

The key point is that the dependence on the vectorial position
:math:`\boldsymbol{r}_{ij}` is split into a factor that contains information about the
orientation of this vector, namely :math:`Y_{lm}(\hat{r}_{ij})`, which is just the
spherical harmonic evaluated at :math:`\hat{r}_{ij}`, and a remaining part that captures
the dependence on the distance of atom :math:`j` from the center atom :math:`i`, namely
:math:`I_{nl}(r_{ij})`, which we shall call the radial integral. The radial integral
class computes and outputs this radial part :math:`I_{nl}(r_{ij})`. Since the angular
part is just the usual spherical harmonic, this is the part that also depends on the
choice of atomic density :math:`g(r)`, as well as the radial basis :math:`R_{nl}(r)`. In
the following, for users only interested in a specific type of density, we provide the
explicit expressions of :math:`I_{nl}(r)` for the Delta and Gaussian densities, followed
by the general expression.

Delta Densities
~~~~~~~~~~~~~~~

Here, we consider the especially simple special case where the atomic density function
:math:`g(\boldsymbol{r}) = \delta(\boldsymbol{r})`. Then:

.. math::

   \begin{aligned}
      \langle nlm | g;\boldsymbol{r}_{ij} \rangle & = \int \mathrm{d}^3\boldsymbol{r}
      R_{nl}(r)Y_{lm}(\hat{r})g(\boldsymbol{r} - \boldsymbol{r}_{ij}) \\ & = \int
      \mathrm{d}^3\boldsymbol{r} R_{nl}(r)Y_{lm}(\hat{r})\delta(\boldsymbol{r} -
      \boldsymbol{r}_{ij}) \\ & = R_{nl}(r) Y_{lm}(\hat{r}_{ij}) =
      B_{nlm}(\boldsymbol{r}_{ij}).
   \end{aligned}

Thus, in this particularly simple case, the radial integral is simply the radial basis
function evaluated at the pair distance :math:`r_{ij}`, and we see that the integrals
have indeed the form presented above.

Gaussian Densities
~~~~~~~~~~~~~~~~~~

Here, we consider another popular use case, where the atomic density function is a
Gaussian. In rascaline, we use the convention

.. math::

   g(r) = \frac{1}{(\pi \sigma^2)^{3/4}}e^{-\frac{r^2}{2\sigma^2}}.

The prefactor was chosen such that the “L2-norm” of the Gaussian

.. math::

   \begin{aligned}
      \|g\|^2 = \int \mathrm{d}^3\boldsymbol{r} |g(r)|^2 = 1,
   \end{aligned}

but does not affect the following calculations in any way. With these conventions, it
can be shown that the integral has the desired form

.. math::

   \begin{aligned}
      \langle nlm | g;\boldsymbol{r}_{ij} \rangle & = \int \mathrm{d}^3\boldsymbol{r}
      R_{nl}(r)Y_{lm}(\hat{r})g(\boldsymbol{r} - \boldsymbol{r}_{ij}) \\ & =
      Y_{lm}(\hat{r}_{ij}) \cdot I_{nl}(r_{ij})
   \end{aligned}

with

.. math::

   I_{nl}(r_{ij}) = \frac{1}{(\pi \sigma^2)^{3/4}}4\pi e^{-\frac{r_{ij}^2}{2\sigma^2}}
   \int_0^\infty \mathrm{d}r r^2 R_{nl}(r) e^{-\frac{r^2}{2\sigma^2}}
   i_l\left(\frac{rr_{ij}}{\sigma^2}\right),

where :math:`i_l` is a modified spherical Bessel function. The first factor, of course,
is just the normalization factor of the Gaussian density. See the next two subsections
for a derivation of this formula.

Derivation of the General Case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now derive an explicit formula for radial integral that works for any density. Let
:math:`g(r)` be a generic spherically symmetric density function. Our goal will be to
show that

.. math::

   \langle nlm | g;\boldsymbol{r}_{ij} \rangle = Y_{lm}(\hat{r}_{ij}) \left[2\pi
   \int_0^\infty \mathrm{d}r r^2 R_{nl}(r) \int_{-1}^1 \mathrm{d}(\cos\theta)
   P_l(\cos\theta) g(\sqrt{r^2+r_{ij}^2-2rr_{ij}\cos\theta}) \right]

and thus we have the desired form :math:`\langle nlm | g;\boldsymbol{r}_{ij} \rangle =
Y_{lm}(\hat{r}_{ij}) I_{nl}(r_{ij})` with

.. math::

   \begin{aligned}
      I_{nl}(r_{ij}) = 2\pi \int_0^\infty \mathrm{d}r r^2 R_{nl}(r) \int_{-1}^1
      \mathrm{d}u P_l(u) g(\sqrt{r^2+r_{ij}^2-2rr_{ij}u}),
   \end{aligned}

where :math:`P_l(x)` is the :math:`l`-th Legendre polynomial.

Derivation of the explicit radial integral for Gaussian densities
-----------------------------------------------------------------

Denoting by :math:`\theta(\boldsymbol{r},\boldsymbol{r}_{ij})` the angle between a
generic position vector :math:`\boldsymbol{r}` and the vector
:math:`\boldsymbol{r}_{ij}`, we can write

.. math::

   \begin{aligned}
      g(\boldsymbol{r}- \boldsymbol{r}_{ij}) & = \frac{1}{(\pi
      \sigma^2)^{3/4}}e^{-\frac{(\boldsymbol{r}- \boldsymbol{r}_{ij})^2}{2\sigma^2}} \\
      & = \frac{1}{(\pi
      \sigma^2)^{3/4}}e^{-\frac{(r_{ij})^2}{2\sigma^2}}e^{-\frac{(\boldsymbol{r}^2-
      2\boldsymbol{r}\boldsymbol{r}_{ij})}{2\sigma^2}},
   \end{aligned}

where the first factor no longer depends on the integration variable :math:`r`.

Analytical Expressions for the GTO Basis
----------------------------------------

While the above integrals are hard to compute in general, the GTO basis is one of the
few sets of basis functions for which many of the integrals can be evaluated
analytically. This is also useful to test the correctness of more numerical
implementations.

The primitive basis functions are defined as

.. math::

   \begin{aligned}
       R_{nl}(r) = R_n(r) = r^n e^{-\frac{r^2}{2\sigma_n^2}}
   \end{aligned}

In this form, the basis functions are not yet orthonormal, which requires an extra
linear transformation. Since this transformation can also be applied after computing the
integrals, we simply evaluate the radial integral with respect to these primitive basis
functions.

Real Space Integral for Gaussian Densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now evaluate

.. math::

   \begin{aligned}
       I_{nl}(r_{ij}) & = \frac{1}{(\pi \sigma^2)^{3/4}}4\pi
       e^{-\frac{r_{ij}^2}{2\sigma^2}} \int_0^\infty \mathrm{d}r r^2 R_{nl}(r)
       e^{-\frac{r^2}{2\sigma^2}} i_l\left(\frac{rr_{ij}}{\sigma^2}\right) \\ & =
       \frac{1}{(\pi \sigma^2)^{3/4}}4\pi e^{-\frac{r_{ij}^2}{2\sigma^2}} \int_0^\infty
       \mathrm{d}r r^2 r^n e^{-\frac{r^2}{2\sigma_n^2}} e^{-\frac{r^2}{2\sigma^2}}
       i_l\left(\frac{rr_{ij}}{\sigma^2}\right),
   \end{aligned}

the result of which can be conveniently expressed using :math:`a=\frac{1}{2\sigma^2}`,
:math:`b_n = \frac{1}{2\sigma_n^2}`, :math:`n_\mathrm{eff}=\frac{n+l+3}{2}` and
:math:`l_\mathrm{eff}=l+\frac{3}{2}` as

.. math::

   \begin{aligned}
       I_{nl}(r_{ij}) = \frac{1}{(\pi \sigma^2)^{3/4}} \cdot
       \pi^{\frac{3}{2}}\frac{\Gamma\left(n_\mathrm{eff}\right)}{\Gamma\left(l_\mathrm{eff}\right)}\frac{(ar_{ij})^l}{(a+b)^{n_\mathrm{eff}}}M\left(n_\mathrm{eff},l_\mathrm{eff},\frac{a^2r_{ij}^2}{a^2+b^2}\right),
   \end{aligned}

where :math:`M(a,b,z)` is the confluent hypergeometric function (hyp1f1).
