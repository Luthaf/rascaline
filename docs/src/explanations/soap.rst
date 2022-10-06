.. _theory:

What is SOAP
============

The mathematics behind the analytical radial integrals, and Cartesian gradients,
of the SOAP descriptor [#one]_ is described below.

.. TODO link to an appropriate publication once the below is incorporated there


From the density to the density expansion
-----------------------------------------

Density expansion:

.. we have to get the latex definitions out of the way first

.. math::

   \newcommand{\bvec}[1]{\mathbf{#1}}
   \newcommand{\CHF}[3]{{}_{1}F_{1}\left(#1 ,#2, #3 \right)}
   \newcommand{\CHLF}[2]{{}_{0}F_{1}\left(#1,#2\right)}
   \newcommand{\GA}[1]{\Gamma\left(#1\right)}
   \newcommand{\rij}{r_{ij}}
   \newcommand{\phiij}{\phi_{ij}}
   \newcommand{\uij}{u_{ij}}
   \newcommand{\thetaij}{\theta_{ij}}
   \newcommand{\br}{\mathbf{r}}
   \newcommand{\bhr}{\hat{\mathbf{r}}}
   \newcommand{\bm}[1]{\mathbf{#1}}
   \newcommand{\bu}{\mathbf{u}}
   \newcommand{\Rhat}{{\hat{R}}}
   \newcommand{\bk}{\mathbf{k}}
   \newcommand{\bK}{\mathbf{K}}
   \newcommand{\bt}{\mathbf{t}}
   \newcommand{\by}{\mathbf{y}}
   \newcommand{\bw}{\mathbf{w}}
   \newcommand{\bz}{\mathbf{z}}
   \newcommand{\bx}{\mathbf{x}}
   \newcommand{\bPsi}{\mathbf{\Psi}}
   \newcommand{\bPhi}{\mathbf{\Phi}}
   \newcommand{\dbr}{\textrm{d}\br}
   \newcommand{\dint}{\textrm{d}}
   \newcommand{\drhat}{\textrm{d}\hat{R}}
   \newcommand{\cald}{\mathcal{D}}
   \newcommand{\CX}{\mathcal{X}}
   \newcommand{\calk}{\mathcal{K}}
   \newcommand{\CA}{\mathcal{A}}
   \newcommand{\CB}{\mathcal{B}}
   \newcommand{\loss}{\ell}
   \newcommand{\diag}{\operatorname{diag}}
   % Stuff that was in the physics package (but mathjax can't use, obvs)
   \newcommand{\braket}[2]{\left\langle #1 \middle| #2 \right \rangle}
   \newcommand{\dd}{\mathrm{d}\;}
   \newcommand{\norm}[1]{\left\| #1 \right\|}
   \newcommand{\dv}[2]{\frac{\mathrm{d}\, #1}{\mathrm{d}\, #2}}
   \newcommand{\grad}{\nabla}
   \newcommand{\iket}[2]{\ket{#1}_{#2}}
   \newcommand{\ibraket}[3]{\braket{#1}{#2}_{#3}}
   \newcommand{\CN}{\mathcal{N}}
   \newcommand{\gaus}[2]{\CN_{#1}\left(#2\right)}

   \rho^{\alpha}_i(\mathbf{r})=\sum_{j\in \alpha} \mathcal{N}_{\sigma}\left(\mathbf{r}-\mathbf{r}_{ij}\right) =
   \sum_{nlm} \braket{\alpha n \ell m}{\mathcal{X}_i} B_{nlm}(\mathbf{r}),

where :math:`\alpha` is a tag refers to the species of the considered
atoms, :math:`\sum_{j \in \alpha}` defines a sum over the neighbours of
atom :math:`i` of species :math:`\alpha`, :math:`\mathcal{N}_{\sigma}` a
Gaussian centered around :math:`0` and variance :math:`\sigma^2`,
:math:`B_{nlm}(\mathbf{r}) = R_n(r) Y_{\ell}^m(\hat{\mathbf{r}})`
defines a complete orthonormal basis set, :math:`Y_{\ell}^m` a spherical
harmonic (SH), :math:`r=\norm{\mathbf{r}}` and
:math:`\hat{\mathbf{r}}=\mathbf{r}/r`.

The following derivation aims at deriving expressions for the
coefficients of the density expansion and their derivative with respect
to atomic coordinate for several basis sets.

Density coefficients: angular integration
-----------------------------------------

Spherical harmonics are the only orthonormal basis set of :math:`S^2` so
this part should apply except for real space basis.

We use the orthonormality of the basis set to compute the expressiont
for the density coefficients and express the resulting integral over
:math:`\rm I\!R^3` in spherical coordinates using
:math:`\norm{\mathbf{r}-\mathbf{r}_{ij}}^2=\mathbf{r}^2+\mathbf{r}_{ij}^2-2\norm{\mathbf{r}}\norm{\mathbf{r}_{ij}}\cos{\theta}`:

.. math::

   \begin{split}
   \braket{\alpha n \ell m}{\mathcal{X}_i}=& \sum_{j \in \alpha} c^{ij}_{n \ell m} = \sum_{j \in \alpha} \int_{\rm I\!R^3} \exp\left[-a\left(\mathbf{r}-\mathbf{r}_{ij}\right)^2\right]B_{nlm}(\mathbf{r})\\
   =&\sum_{j \in \alpha} \int_{0}^{\infty}r^2  \exp\left[-a\left(r^2+r_{ij}^2\right)\right] g_n(r) \int_{-1}^{1}\mathrm{d}\left(\cos{\theta}\right) \\
   & \int_0^{2\pi}\mathrm{d}\phi \exp\left[2arr_{ij}\cos{\theta}\right]Y_{\ell}^{m}\left(\hat{\mathrm{R}}\hat{\bm{q}}\right),\\
   \end{split}

where
:math:`\hat{\mathrm{R}} = \hat{\mathrm{R}}_{ZYZ}\left(\alpha_{ij},\beta{ij},0\right)`
is the ZYZ-Euler matrix that rotate :math:`\hat{e}_z` onto
:math:`\bm{q}_{ij}`, :math:`a=\frac{1}{2\sigma^2}`. Note that global
normalization constants are omitted because of a normalization at the
end.

| We use the following convention for the SH

  .. math:: Y_{\ell}^{m}\left(\hat{\mathbf{r}}\right)=Y_{\ell}^{m}\left(\theta,\phi\right)=A_{\ell}^{m}e^{im\phi}P^{m}_{\ell}\left(\cos\theta\right),

where
:math:`A_{\ell}^{m} =\sqrt{\frac{(\ell-m)!(2l+1)}{4\pi(\ell+m)!}}` and
:math:`\hat{q}` is the direction vector defined by the angle
:math:`\theta` and :math:`\phi`. Note that the phase factor
:math:`(-1)^\ell` is included in the definition of the Associated
Legendre Polynomials (ALPs). The set of spherical harmonics is
calculated in the
| **librascal/src/math/spherical_harmonics.cc** file, which is provided
with explanations. The total number of SH in the set is
:math:`(l_{max} +1)^2`.

The integration over the angular part yields

.. math::

   \begin{split}
   \int_{-1}^{1}\mathrm{d}\left(\cos{\theta}\right) \exp\left[arr_{ij}\cos{\theta}\right] \int_0^{2\pi}\mathrm{d}\phi Y_{\ell}^{m}\left(\hat{\mathrm{R}}\left(\alpha_{ij},\beta{ij},0\right)\hat{\bm{q}}\right) =& 4\pi Y_\ell^m \left(\beta_{ij},\alpha_{ij}\right) \mathsf{i}_{\ell}\left(arr_{ij}\right), \\
   =& Y_\ell^m \left(\hat{\mathbf{r}}_{ij}\right) \mathsf{i}_{\ell}\left(arr_{ij}\right),
   \end{split}

where :math:`\mathsf{i}_{\ell}` stands for the modified spherical
Bessel function of the first kind. The intermediate steps are detailed
in the following paragraphs.

Integration over :math:`\phi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integration over the polar angle cancels out all orders of :math:`m`
from the SH

.. math:: \int_0^{2\pi}\mathrm{d}\phi Y_{\ell}^{m}\left(\theta,\phi\right) = \sqrt{\pi\left(2l+1\right)}\mathrm{P}_{\ell}^{m}\left(\cos{\theta}\right) \delta_{m0},

since

.. math::

   \begin{split}
   \int_0^{2\pi}\mathrm{d}\phi \exp\left[im\phi\right] = 2\pi \delta_{0m}.
   \end{split}

Nevertheless, the rotation of the spherical harmonic breaks down into a
linear combination of spherical harmonics. The coefficents are the
entries of the Wigner D-matrix constructed from the Euler angles of the
rotation matrix :math:`\hat{R}`.

.. math::

   \begin{split}
   Y_{\ell}^{m}\left(\hat{\mathrm{R}}\hat{\mathbf{r}}\right) = & \sum_{m'=-\ell}^{\ell} \mathrm{D}_{mm'}^\ell\left(\hat{R}\,\right) Y_{\ell}^{m}\left(\hat{\mathbf{r}}\right), \\
   \mathrm{D}_{m0}^\ell\left(\alpha,\beta,\gamma\right) =& \sqrt{\frac{4\pi}{2l+1}} Y_l^m\left(\beta,\alpha\right). \\
   \end{split}

Thus, the polar integral over the rotated SH simplifies into

.. math::

   \begin{split}
   \int_0^{2\pi}\mathrm{d}\phi\, Y_{\ell}^{m}\left(\hat{\mathrm{R}}\hat{\mathbf{r}}\right) =& \sum_{m'=-\ell}^{\ell} \mathrm{D}_{mm'}^\ell\left(\alpha_{ij},\beta_{ij},0\right) \sqrt{\pi(2l+1)} P_{\ell}^{m'}\left(\cos{\theta}\right) \delta_{m'0} \\
   =& 2\pi Y_\ell^m \left(\beta_{ij},\alpha_{ij}\right) P_{\ell}^{0}\left(\cos{\theta}\right).
   \end{split}
   \label{eq:int-phi}

Integration over :math:`\theta`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The modified spherical Bessel function of the first kind (MSBF) admit
the following integral representation

.. math:: \mathsf{i}_n\left(z\right)= \frac{1}{2}\int_{-1}^{1}\mathrm{d}x \exp\left(zx\right)\mathrm{P}_{n}^{0}\left(x\right),

which can be shown using the reference relations [1]_ [2]_ [3]_:

.. math::

   \begin{aligned}
   \mathsf{j}_n\left(z\right) =& \frac{(-i)^n}{2}\int_{-1}^{1}\mathrm{d}x \exp\left[izx\right]P_{n}^{0}\left(x\right), \label{eq:bessel-1}\\
   \mathsf{i}_n\left(z\right)=& (-i)^{n} \mathsf{j}_n\left(iz\right), \label{eq:bessel-3}\\
   \mathsf{i}_n\left(z\right)=& (-1)^{n} \mathsf{i}_n\left(-z\right), \label{eq:bessel-4}\end{aligned}

.. [1] http://dlmf.nist.gov/10.54.E2

.. [2] http://dlmf.nist.gov/10.47.E12

.. [3] http://dlmf.nist.gov/10.47.E16

:math:`j_n` is the spherical Bessel function of the first kind. The
integral over the polar angle is then given by

.. math::

   \begin{split}
   \int_{-1}^{1}\mathrm{d}\left(\cos{\theta}\right) \exp\left[2arr_{ij}\cos{\theta}\right]P_{\ell}^{0}\left(\cos{\theta}\right) =& 2 \mathsf{i}_{\ell}(2arr_{ij}).
   \end{split}

Density coefficients: Radial integration
----------------------------------------

Summing up the results from the previous section:

.. math:: c^{ij}_{n\ell m} = 4\pi Y_{\ell}^m(\hat{\mathbf{r}}_{ij}) \exp\left[-ar^2_{ij}\right] \underbrace{\int_0^\infty \dd{r} r^2 R_n(r) e^{-ar^2} \mathsf{i}_{\ell}\left(2a r r_{ij}\right)}_{=\text{I}_{n\ell}^{ij}} ,

we identify :math:`\text{I}_{n\ell}^{ij}` as the last term to simplify
for particular choices of radial basis functions.

GTO like radial basis
~~~~~~~~~~~~~~~~~~~~~

The Gaussian Type Orbital radial basis is defined

.. math:: R^{GTO}_{n}(r) = \mathcal{N}_n\ r^{n} \exp[-br^2],

where :math:`b=\frac{1}{2\sigma_n^2}`,
:math:`\sigma_n = (r_\text{cut}-\delta r_\text{cut}) \max(\sqrt{n},1)/n_\text{max}`
and the normalization factor is given by

.. math:: \mathcal{N}_n^2 = \frac{2(1)}{\sigma_n^{2n + 3}\Gamma(n + 3/2)}.

The overlap between GTO radial basis is:

.. math:: \int_0^\infty R^{GTO}_{n}(r) R^{GTO}_{n^\prime}(r) \dd{r}= 2 \left(\frac{1}{2 \sigma_{n}^2}+\frac{1}{2 \sigma_{n^\prime}^2} \right)^{-\frac{1}{2} (3+n+n^\prime)} \Gamma(\frac{3+n+n^\prime}{2})

This equals what we use in the implementation

.. math:: \int_0^\infty R^{GTO}_{n}(r) R^{GTO}_{n^\prime}(r) \dd{r}= N_n N_{n^\prime} \left(\frac{1}{2 \sigma_{n}^2}+\frac{1}{2 \sigma_{n^\prime}^2} \right)^{-\frac{1}{2} (3+n+n^\prime)} \Gamma(\frac{3+n+n^\prime}{2})

The radial integral becomes

.. math::

   I^{ij\,\text{GTO}}_{nl}= \mathcal{N}_n \frac{\sqrt{\pi}}{4} \frac{\GA{\frac{n+\ell+k+3}{2}}}{\GA{\ell+\frac{3}{2}}}a^\ell \rij^\ell(a+b)^{-\frac{n+k+\ell+3}{2}}  \CHF{\frac{n+\ell+k+3}{2}}{\ell+\frac{3}{2}}{\frac{a^2 \rij^2}{a+b}},
   \label{eq:rad-int-gto-1}

which yields the following expression for the neighbour contribution

.. math::

    c^{ij\,\text{GTO}}_{n\ell m}=& (\pi)^{\frac{3}{2}} \mathcal{N}_n \frac{\GA{\frac{n+\ell+3}{2}}}{\GA{\ell+\frac{3}{2}}} (a+b)^{-\frac{n+\ell+3}{2}}  \\
    & Y_{\ell}^m(\bhr_{ij}) \exp\left[-ar^2_{ij}\right]   (a\rij)^\ell  \CHF{\frac{n+\ell+3}{2}}{\ell+\frac{3}{2}}{\frac{a^2 \rij^2}{a+b}}.
    \label{eq:density-gto}

where :math:`\Gamma` is the Gamma function, and :math:`{}_1F_1` is the
confluent hypergeometric function of the first kind.

| The neighbour contribution is calculated in
| file **librascal/src/representations/
  representation_manager_spherical_expansion.hh**,
| function **compute_neighbour_contribution**, line 338.

The steps of the derivation are detailed in the next paragraph.

Analytic radial integral
^^^^^^^^^^^^^^^^^^^^^^^^

We write an integral representation of the confluent hypergeometric
function :math:`\CHF{a}{b}{z}` (CHF) in
terms of MSBF:

.. math::

   \CHF{a}{\ell+\frac{3}{2}}{x} = \frac{2x^{-\frac{\ell}{2}}}{\sqrt{\pi}}\frac{\GA{\ell+\frac{3}{2}}}{\GA{a}}\int_0^\infty e^{-t} t^{a-1-\frac{\ell}{2}} \mathsf{i}_{\ell}(2\sqrt{xt})\dd{t},
   \label{eq:chf-int}

using these relations [4]_ [5]_ [6]_

.. math::
   \begin{align}
   \CHF{a}{b}{z} = & \frac{1}{\GA{a}} \int_0^\infty e^{-t}t^{a-1}\CHLF{b}{zt}\dd{t},\\
   I_l(z) =& \frac{(\frac{z}{2})^{\ell}}{\GA{l+1}} \CHLF{\ell+1}{\frac{z^2}{4}},\\
   \mathsf{i}_{\ell}(z) =& \sqrt{\frac{\pi}{2z}}I_{\ell+1/2}(z),\\
   \mathsf{i}_{\ell}(z) =& \sqrt{\frac{\pi}{4}}\frac{(\frac{z}{2})^{\ell}}{\GA{\ell+\frac{3}{2}}} \CHLF{\ell+\frac{3}{2}}{\frac{z^2}{4}},\\
   \CHLF{\ell+\frac{3}{2}}{xt}=& \sqrt{\frac{4}{\pi}}\GA{\ell+\frac{3}{2}} x^{-\frac{\ell}{2}}t^{-\frac{\ell}{2}}\mathsf{i}_{\ell}(2\sqrt{xt}),
   \end{align}

.. [4]

   http://functions.wolfram.com/HypergeometricFunctions/Hypergeometric1F1/07/01/01/0002/
   http://dlmf.nist.gov/16.5.E3

.. [5] https://en.wikipedia.org/wiki/Generalized_hypergeometric_function#The_series_0F1
.. [6] http://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html


where :math:`I_\ell` is the modified Bessel function and
:math:`\CHLF{b}{z}` is the limit conflent
hypergeometric function.

The module for calculating
:math:`\CHF{..}{..}{..}` is located in
**librascal/src/math/hyp1f1.hh**.

The radial integral with GTO radial basis function is:

.. math::

   I^{ij\,\text{GTO}}_{nl}=\int_0^\infty \dd{r} r^{2+k} g^{\text{GTO}}_n(r) e^{-\frac{r^2}{2\sigma^2}} \mathsf{i}_{\ell}\left(r r_{ij} / \sigma^2\right) = \mathcal{N}_n \int_0^\infty \mathrm{d}r r^{2+k+n}  e^{-r^2(a+b)} \mathsf{i}_{\ell}\left(2a r r_{ij}\right),
       \label{eq:rad-int-gto-0}

with :math:`k` an additional power of :math:`r` that will be non zero
for the derivative. We partially identify the terms between
`[eq:chf-int] <#eq:chf-int>`__ and
`[eq:rad-int-gto-0] <#eq:rad-int-gto-0>`__:

.. math::

   \begin{aligned}
       t =& r^2(a+b),\\
       \dd{t} =& 2 r \dd{r} (a+b),\\
       x = & \frac{a^2 r_{ij}^2}{a+b},\end{aligned}

to change the integrand of the radial integral

.. math::

   I^{ij\,\text{GTO}}_{nl}= \mathcal{N}_n \int_0^\infty \frac{\dd{t}}{2(a+b)} (a+b)^{-\frac{n+k+1}{2}} t^{\frac{n+k+1}{2}}  e^{-t} \mathsf{i}_{\ell}\left(2\sqrt{xt}\right),
       \label{eq:rad-int-gto-01}

and identify the last term

.. math::

   \begin{aligned}
       a =& \frac{n+\ell+k+3}{2}.\end{aligned}

Numerical Integration of the Radial Integral
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The numerical integration does not rely on a specific form of the radial
basis

.. math:: \text{I}_{n\ell}^{ij} = \sum_{k=1}^{K} \omega_k  r_k^2 R_n(r_k) e^{-ar_k^2} \mathsf{i}_{\ell}\left(2a r_k r_{ij}\right),

where the :math:`\omega_k` are the quadrature weights evaluated at the
quadrature nodes :math:`r_k`. Depending on the quadrature rule, the
following shifting formula is useful,

.. math:: \int_a^b f(x)\,\dd{x} \approx \frac{b-a}{2} \sum_{i=1}^n w_i f\left(\frac{b-a}{2}x_i + \frac{a+b}{2}\right).

Discrete Variable Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the special case of the the DVR radial basis [7]_ with Gauss-Legendre
quadrature rule, the radial integral simplifies into:

.. math:: \text{I}_{n\ell}^{ij} = \frac{r_c}{2} \sqrt{\omega_n} x_n^2 e^{-ax_n^2} \mathsf{i}_{\ell}\left(2a x_n r_{ij}\right),

where :math:`x_n=\frac{r_c}{2}r_n+\frac{r_c}{2}`.

.. [7]

   Light, J. C., & Carrington, T. (2007). Discrete-Variable Representations and their Utilization (pp. 263–310). John Wiley & Sons, Ltd.
   https://doi.org/10.1002/9780470141731.ch4

Gradient of the density coefficients with respect to the Cartesian coordinates
------------------------------------------------------------------------------

The density coefficients can be split into two parts: one that depends on
the choice of radial basis function (:math:`\text{I}_{n\ell}^{ij}`) and
the rest:

.. math:: c^{ij}_{n\ell m} =  Y_{\ell}^m(\hat{\mathbf{r}}_{ij}) \exp\left[-ar^2_{ij}\right] \text{I}_{n\ell}^{ij} =  D^{ij}_{\ell m} C^{ij} \text{I}_{n\ell}^{ij},

where :math:`C^{ij}` is the Gaussian exponential factor and
:math:`\bar{D}^{ij}_{\ell m} = \bar{Y}_{\ell,m}(\hat{r}_{ij})` is the
spherical harmonic, see eq.
`[eq:real-spherical-harmonics] <#eq:real-spherical-harmonics>`__. Note
the constant factors are omitted.

The following derivations end up with this formula that does not depend
on the radial basis:

.. math::

   \begin{aligned}
       \grad_i\,c^{ij}_{\alpha n \ell m} =& 2a c^{ij}_{\alpha n \ell m} \mathbf{r}_{ij}\nonumber\\
       &{} +  C \bar{D}^{ij}_{\ell m} \cdot \grad_i \text{I}_{n\ell}^{ij}\nonumber\\
       &{} + N_{n \ell}A_{n\ell} B_\ell C \cdot \grad_i\,\bar{D}^{ij}_{\ell,m},\end{aligned}

where
:math:`\grad_i\bar{D}^{ij}_{\ell,m} = \grad_i \bar{Y}_{\ell,m}(\hat{r}_{ij})`
is defined in
`[eq:dbx0,eq:dbx1,eq:dbx2,eq:dby0,eq:y1,eq:dby2,eq:dbz0,eq:dbz1,eq:dbz2] <#eq:dbx0,eq:dbx1,eq:dbx2,eq:dby0,eq:y1,eq:dby2,eq:dbz0,eq:dbz1,eq:dbz2>`__.

Terms common to the different radial basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gaussian
^^^^^^^^

.. math::

   \begin{gathered}
       \dv{C^{ij}}{r_{ij}} = -2ar_{ij}C^{ij}\end{gathered}

Length
^^^^^^

So for the radial terms, we just use the derivatives of the radius
:math:`r_{ij}` wrt the Cartesian coordinates:

.. math::

   \begin{gathered}
       \dv{ r_{ij}}{ \{x_i, y_i, z_i\}} = -\frac{\{x_{ij}, y_{ij}, z_{ij}\}}{r_{ij}}\\
       \grad_i\,r_{ij} = \frac{-\mathbf{r}_{ij}}{r_{ij}}\\
       \text{where }\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i\end{gathered}

Spherical Harmonics
^^^^^^^^^^^^^^^^^^^

The derivative of the spherical harmonic can be expressed in a few
different ways. The versions below are in terms of the original harmonic
with possibly different :math:`m` values. The :math:`z` component is:

.. math::

   \begin{aligned}
       \frac{\partial D_{\ell m}}{\partial z_i} &= \frac{-\sqrt{1-u^2}}{2r}\big(e^{i\phi}\sqrt{(\ell+m)(\ell-m+1)}Y_l^{m-1}(\hat{r})\nonumber\\
           &\qquad\qquad - e^{-i\phi}\sqrt{(\ell-m)(\ell+m+1)}Y_l^{m+1}(\hat{r})\big)\nonumber\\
       &= \frac{-\sin{\theta}}{2r_{ij}}(\cos(m\phi) + i\sin(m\phi)) \\
           &\qquad\qquad \left(\sqrt{(\ell+m)(\ell - m + 1)}\sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m+1)!}{(\ell+m-1)!}}
           P_l^{m-1}(\cos{\theta})\right.\nonumber\\
           &\qquad\qquad\qquad \left. {} - \sqrt{(\ell-m)(\ell + m + 1)}\sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m-1)!}{(\ell+m+1)!}}
               P_l^{m+1}(\cos{\theta})\right)
    \end{aligned}

But remember, we’re actually using the real spherical harmonics:

[eq:real-spherical-harmonics]

.. math::

   \begin{aligned}
       \left.\begin{aligned}
       \bar{Y}_{\ell m}(\hat{r}_{ij}) &= \cos(m\phi) \bar{P}_\ell^m(\cos{\theta})\\
       \bar{Y}_{\ell,-m}(\hat{r}_{ij}) &= \sin(m\phi) \bar{P}_\ell^m(\cos{\theta})
       \end{aligned}\right\}&\text{ for }m > 0\\
       \bar{Y}_{\ell,0}(\hat{r}_{ij}) = \frac{1}{\sqrt{2}} \bar{P}_\ell^0(\cos{\theta})&\end{aligned}

where

.. math:: \bar{P}_\ell^m(\cos{\theta}) = \sqrt{\frac{2\ell + 1}{2\pi}\frac{(\ell - m)!}{(\ell + m)!}}P_\ell^m(\cos{\theta}).

So we can write

.. math::

   \begin{aligned}
       \frac{\partial \bar{D}_{\ell m}}{\partial z_i} &=
       \frac{-\sin\theta}{2r_{ij}}\cos(m\phi)\left(\sqrt{(\ell + m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta)
           - \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right) \label{eq:dbz0}\\
       \frac{\partial \bar{D}_{\ell,-m}}{\partial z_i} &=
       \frac{-\sin\theta}{2r_{ij}}\sin(m\phi)\left(\sqrt{(\ell + m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta)
           - \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right)\label{eq:dbz1}\\
       \frac{\partial \bar{D}_{\ell,0}}{\partial z_i} &=
           \frac{\sin\theta}{r_{ij}}
               \sqrt{\frac{\ell(\ell + 1)}{2}}\bar{P}_\ell^{1}(\cos\theta))\label{eq:dbz2}\end{aligned}

(the last one comes from the identity
:math:`\sqrt{\frac{(\ell+m)!}{(\ell-m)!}}P_\ell^{-m} = (-1)^m \sqrt{\frac{(\ell - m)!}{(\ell + m)!}}P_l^m(\cos\theta)`
with :math:`m=1`).

The :math:`x` component is:

.. math::

   \begin{aligned}
       \frac{\partial \bar{D}_{\ell m}}{\partial x_i} &= \frac{-m\sin\phi}{\sqrt{x_{ij}^2 + y_{ij}^2}} \bar{D}_{\ell,-m} + \frac{\cos\phi \cos\theta}{2r_{ij}}\cos(m\phi)\left(
           \sqrt{(\ell + m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta)\right.\nonumber\\
           &\qquad\qquad\qquad\left. {} - \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right)\label{eq:dbx0}\\
       \frac{\partial \bar{D}_{\ell,-m}}{\partial x_i} &= \frac{m\sin\phi}{\sqrt{x_{ij}^2 + y_{ij}^2}} \bar{D}_{\ell,m} + \frac{\cos\phi \cos\theta}{2r_{ij}}\sin(m\phi)\left(
           \sqrt{(\ell + m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta)\right.\nonumber\\
           &\qquad\qquad\qquad\left. {} - \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right)\label{eq:dbx1}\\
       \frac{\partial \bar{D}_{\ell,0}}{\partial x_i} &=
           \frac{-\cos\phi \cos\theta}{r_{ij}}\sqrt{\frac{\ell(\ell+1)}{2}}\bar{P}_\ell^1(\cos\theta)\label{eq:dbx2}\end{aligned}

and for the :math:`y` component, similarly:

.. math::

   \begin{aligned}
       \frac{\partial \bar{D}_{\ell m}}{\partial y_i} &= \frac{m\cos\phi}{\sqrt{x_{ij}^2 + y_{ij}^2}} \bar{D}_{\ell,-m} + \frac{\sin\phi \cos\theta}{2r_{ij}}\cos(m\phi)\left(
           \sqrt{(\ell + m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta)\right.\nonumber\\
           &\qquad\qquad\qquad\left. {} - \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right)\label{eq:dby0}\\
       \frac{\partial \bar{D}_{\ell,-m}}{\partial y_i} &= \frac{-m\cos\phi}{\sqrt{x_{ij}^2 + y_{ij}^2}} \bar{D}_{\ell,m} + \frac{\sin\phi \cos\theta}{2r_{ij}}\sin(m\phi)\left(
           \sqrt{(\ell + m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta)\right.\nonumber\\
           &\qquad\qquad\qquad\left. {} - \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right)\label{eq:dby1}\\
       \frac{\partial \bar{D}_{\ell,0}}{\partial y_i} &=
           \frac{-\sin\phi \cos\theta}{r_{ij}}\sqrt{\frac{\ell(\ell+1)}{2}}\bar{P}_\ell^1(\cos\theta)\label{eq:dby2}\end{aligned}

The formulæ above have a singularity at the poles for :math:`m \neq 0`,
so use the following identity:

.. math::

   \begin{gathered}
       \frac{m}{\sqrt{x_{ij}^2 + y_{ij}^2}} \begin{pmatrix}\bar{Y}_{\ell, -m}(\hat{r}_{ij})\\
                                                            \bar{Y}_{\ell,  m}(\hat{r}_{ij})\end{pmatrix}
           = \frac{-1}{2z_{ij}}\begin{pmatrix}\sin(m\phi)\\\cos(m\phi)\end{pmatrix}
               \left(\sqrt{(\ell+m)(\ell - m + 1)}\bar{P}_\ell^{m-1}(\cos\theta) \right.\\
               \left. {} + \sqrt{(\ell - m)(\ell + m + 1)}\bar{P}_\ell^{m+1}(\cos\theta)\right)\end{gathered}

to shift the singularity to the equator (:math:`z=0`). In the code
derivatives of spherical harmonics is computed in the
**feat/soap_gradients branch**,
**librascal/src/math/spherical_harmonics.hh**

.. _gto-like-radial-basis-1:

GTO like radial basis
~~~~~~~~~~~~~~~~~~~~~

We rewrite `[eq:rad-int-gto-1] <#eq:rad-int-gto-1>`__

.. math:: I^{ij\,\text{GTO}}_{nl} = N_{n\ell} \cdot A_{n\ell} \cdot B_\ell ,

where :math:`B_{\ell} = r_{ij}^{\ell}`,
:math:`A_{n\ell} = \CHF{\frac{n + \ell + 3}{2}}{\ell+\frac{3}{2}}{\frac{a^2 r_{ij}^2}{a+b}}`,
:math:`N_{n \ell} = \frac{\mathcal{N}_n}{4} a^\ell\left(a+b\right)^{-\frac{n + \ell + 3}{2}}  \frac{\Gamma\left(\frac{n + \ell + 3}{2}\right)}{\Gamma\left(\frac{3}{2} + \ell\right)}`,
:math:`\mathcal{N}_n = \sqrt{\frac{2}{\sigma_n^{2n+3}\Gamma\left(n + \frac{3}{2}\right)}}`.
Note that some constant multiplying factors of :math:`\pi` have been
omitted.

:math:`B_{\ell}`
^^^^^^^^^^^^^^^^

.. math:: \dv{B_\ell}{r_{ij}} = \frac{\ell}{r_{ij}} B_\ell

CHF
^^^

for the hypergeometric term:

.. math::

    \dv{A_{n \ell}}{r_{ij}} = \frac{\frac{n + \ell + 3}{2}}{\left(\ell + \frac{3}{2}\right)}
    \frac{2a^2 r_{ij}}{a+b}
    \CHF{\frac{n + \ell + 5}{2}}{\ell+\frac{5}{2}}{\frac{a^2 r_{ij}^2}{a+b}}

which is not proportional to :math:`A_{n \ell}`, or even to
:math:`A_{n+1,\ell + 1}` – so just recompute it explicitly.

GTO formula for practical computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, putting the radial and angular components together, we get:

.. math::

   \begin{aligned}
    \grad_i\,c^{ij}_{\alpha n \ell m} &= c^{ij}_{\alpha n \ell m}\left(-\frac{\ell}{r_{ij}^2} + 2a\right)\br_{ij}\nonumber\\
    &{} + N_{n \ell}B_\ell C \bar{D}_{\ell m} \cdot \frac{\frac{n + \ell + 3}{2}}{\left(\ell + \frac{3}{2}\right)}
    \frac{2a^2}{a+b}
    \CHF{\frac{n + \ell + 5}{2}}{\ell+\frac{5}{2}}{\frac{a^2 r_{ij}^2}{a+b}} \bvec{r}_{ij}\nonumber\\
    &{} + N_{n \ell}A_{n\ell} B_\ell C \cdot \nabla_i\,\bar{D}_{\ell,m}
   \end{aligned}

where the gradient of the spherical harmonic has already been computed
separately using the equations above.

| Gradient of the coefficients is calculated in **feat/soap_gradients**
  branch,
| file
  **librascal/src/representations/representation_manager_spherical_expansion.hh**,
| function **compute_neighbour_derivative**, line 420.

Numerical Integration
~~~~~~~~~~~~~~~~~~~~~

Using the recurrence relation of the MSBF [6]_:

.. math:: \dv{\mathsf{i}_{\ell}(x)}{x} = \frac{1}{2\ell+1}[\ell\mathsf{i}_{\ell-1}(x)+(\ell+1)\mathsf{i}_{\ell+1}(x)],

the gradient of the radial integral becomes:

.. math:: \grad_i \text{I}_{n\ell}^{ij} = -\frac{2a}{2\ell+1}\sum_{k=1}^{K} \omega_k  r_k^3 R_n(r_k) e^{-ar_k^2} [\ell\mathsf{i}_{\ell-1}(2a r_k r_{ij})+(\ell+1)\mathsf{i}_{\ell+1}(2a r_k r_{ij})] \hat{\mathbf{r}}_{ij}.

In the case of the DVR radial basis:

.. math:: \text{I}_{n\ell}^{ij} = -\frac{2a\sqrt{\omega_n}}{2\ell+1}\frac{r_c}{2}  x_n^3 e^{-ax_n^2} [\ell\mathsf{i}_{\ell-1}(2a x_n r_{ij})+(\ell+1)\mathsf{i}_{\ell+1}(2a x_n r_{ij})] \hat{\mathbf{r}}_{ij},

where :math:`x_n=\frac{r_c}{2}r_n+\frac{r_c}{2}`.

Beyond SOAP
-----------

SOAP can be seen as one of the simplest members of a hierarchy of "density correlation features", 
that are obtained by appropriately symmetrizing tensor products of the neighbor density. 
The formalism, first introduced in Ref.  [#two]_, leads to formulas to evaluate discretized
versions of these features as a combination of the density expansion coefficients, and includes
also *equivariant* features, that transform as spherical harmonics under rotation [#three]_.
A comparatively simple expression to compute these higher-order features iteratively, based
on an angular momentum combination relation has been discussed in Ref. [#four]_, as part
of the N-body iterative contraction of equivariants (NICE) framework.
Even though these higher-order features are not the main focus of ``librascal``, you can find 
some utilities that compute them starting from the expansion coefficients. These are 
part of the of the ``bindings/rascal/utils/cg_utils.py``, and are demonstrated and tersely
documented in the example notebooks ``examples/equivariant_demo.ipynb`` and
``examples/nice_demo.ipynb``.


.. [#one] A. P. Bartók, R. Kondor, and G. Csányi (2013)
        On representing chemical environments. 
        Physical Review B 87(18), 184115. https://doi.org/10.1103/PhysRevB.87.184115
        
.. [#two] M. J. Willatt, F. Musil, and M. Ceriotti (2019)
        Atom-density representations for machine learning.
        Journal of Chemical Physics 150(15), 154110. https://doi.org/10.1063/1.5090481

.. [#three] A. Grisafi, D. M. Wilkins, G. Csányi, and M. Ceriotti (2018)
        Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic Systems.
        Physical Review Letters 120(3), 036002. https://doi.org/10.1103/PhysRevLett.120.036002

.. [#four] J. Nigam, S. Pozdnyakov, and M. Ceriotti (2020)
        Recursive evaluation and iterative contraction of N -body equivariant features.
        J. Chem. Phys. 153(12), 121101. https://doi.org/10.1063/5.0021116

        


