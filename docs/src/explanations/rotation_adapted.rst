Rotation-Adapted Features
=========================

Equivariance
------------

Descriptors like SOAP are translation, rotation, and permutation invariant.
Indeed, such invariances are extremely useful if one wants to learn an invariant
target (e.g., the energy). Being already encoded in the descriptor, the learning
algorithm does not have to learn such a physical requirement.

The situation is different if the target is not invariant. For example, one may
want to learn a dipole. The dipole rotates with a rotation of the molecule, and
as such, invariant descriptors do not have the required symmetries for this
task.

Instead, one would need a rotation equivariant descriptor. Rotation equivariance
means that, if we first rotate the system and compute the descriptor, we obtain
the same result as first computing the descriptor and then applying the
rotation, i.e., the descriptor behaves correctly upon rotation operations.
Denoting a system as :math:`A`, the function computing the descriptor as
:math:`f(\cdot)`, and the rotation operator as :math:`\hat{R}`, rotation
equivariance can be expressed as:

.. math::
   :name: eq:equivariance

   f(\hat{R} A) = \hat{R} f(A)

Of course, invariance is a special case of equivariance.


Rotation Equivariance of the Spherical Expansion
------------------------------------------------

The spherical expansion is a rotation equivariant descriptor.
Let's consider the expansion coefficients of :math:`\rho_i(\mathbf{r})`.
We have:

.. math::

    \hat{R} \rho_i(\mathbf{r}) &= \sum_{nlm} c_{nlm}^{i} R_n(r) \hat{R} Y_l^m(\hat{\mathbf{r}}) \nonumber \\
    &= \sum_{nlmm'} c_{nlm}^{i} R_n(r) D_{m,m'}^{l}(\hat{R}) Y_l^{m'}(\hat{\mathbf{r}}) \nonumber \\
    &= \sum_{nlm} \left( \sum_{m'} D_{m',m}^l(\hat{R}) c_{nlm'}^{i}\right) B_{nlm}(\mathbf{r}) \nonumber

and noting that :math:`Y_l^m(\hat{R} \hat{\mathbf{r}}) = \hat{R}
Y_l^m(\hat{\mathbf{r}})` and :math:`\hat{R}r = r`, equation :ref:`(1)
<eq:equivariance>` is satisfied and we conclude that the expansion coefficients
:math:`c_{nlm}^{i}` are rotation equivariant. Indeed, each :math:`c_{nlm}^{i}`
transforms under rotation as the spherical harmonics
:math:`Y_l^m(\hat{\mathbf{r}})`.

Using the Dirac notation, the coefficient :math:`c_{nlm}^{i}` can be expressed
as :math:`\braket{nlm\vert\rho_i}`. Equivalently, and to stress the fact that
this coefficient describes something that transforms under rotation as a
spherical harmonics :math:`Y_l^m(\hat{\mathbf{r}})`, it is sometimes written as
:math:`\braket{n\vert\rho_i;lm}`, i.e., the atomic density is "tagged" with a
label that tells how it transforms under rotations.


Completeness Relations of Spherical Harmonics
---------------------------------------------

Spherical harmonics can be combined together using rules coming from standard
theory of angular momentum:

.. math::
    :name: eq:cg_coupling

    \ket{lm} \propto \ket{l_1 l_2 l m} = \sum_{m_1 m_2} C_{m_1 m_2 m}^{l_1 l_2 l} \ket{l_1 m_1} \ket{l_2 m_2}

where :math:`C_{m_1 m_2 m}^{l_1 l_2 l}` is a Clebsch-Gordan (CG) coefficient.

Thanks to the one-to-one correspondence (under rotation) between
:math:`c_{nlm}^{i}` and :math:`Y_l^m`, :ref:`(2) <eq:cg_coupling>` means that
one can take products of two spherical expansion coefficients (which amounts to
considering density correlations), and combine them with CG coefficients to get
new coefficients that transform as a single spherical harmonics. This process is
known as coupling, from the uncoupled basis of angular momentum (formed by the
product of rotation eigenstates) to a coupled basis (a single rotation
eigenstate).

One can also write the inverse of :ref:`(2) <eq:cg_coupling>`:

.. math::
    :name: eq:cg_decoupling

    \ket{l_1 m_1} \ket{l_2 m_2} = \sum_{l m} C_{m_1 m_2 m}^{l_1 l_2 l m} \ket{l_1 l_2 l m}

that express the product of two rotation eigenstates in terms of one. This
process is known as decoupling.

Example: :math:`\lambda`-SOAP
-----------------------------

A straightforward application of :ref:`(2) <eq:cg_coupling>` is the construction
of :math:`\lambda`-SOAP features. Indeed, :math:`\lambda`-SOAP was created in
order to have a rotation and inversion equivariant version of the 3-body density
correlations. The :math:`\lambda` represents the degree of a spherical
harmonics, :math:`Y_{\lambda}^{\mu}(\hat{\mathbf{r}})`, and it indicates that
this descriptor can transform under rotations as a spherical harmonics, i.e., it
is rotation equivariant.

It is then obtained by considering two expansion coefficients of the atomic
density, and combining them with a CG iteration to a coupled basis, as in
:ref:`(2) <eq:cg_coupling>`. The :math:`\lambda`-SOAP descriptor is then:

.. math::

    \braket{n_1 l_1 n_2 l_2\vert\overline{\rho_i^{\otimes 2}, \sigma, \lambda \mu}} =
    \frac{\delta_{\sigma, (-1)^{l_1 + l_2 + \lambda}}}{\sqrt{2 \lambda + 1}}
    \sum_{m} C_{m (\mu-m) \mu}^{l_1 l_2 \lambda} c_{n_1 l_1 m}^{i} c_{n_2 l_2 (\mu - m)}^{i}

where we have assumed complex spherical harmonics coefficients.
