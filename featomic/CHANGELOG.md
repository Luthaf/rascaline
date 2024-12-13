# Changelog

All notable changes to featomic are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/featomic/)

<!-- Possible sections for each package:

### Added

### Fixed

### Changed

### Removed
-->


### Added

- Multiple atomistic features calculators with a native implementation:
    - SOAP spherical expansion, radial spectrum, power spectrum and spherical
      expansion for pairs of atoms;
    - LODE spherical expansion;
    - Neighbor list;
    - Sorted distances vector;
    - Atomic composition.

- All the calculator outputs are stored in
  [metatensor's](https://docs.metatensor.org/) `TensorMap` objects. This allow
  to both store the features in a very sparse format, saving memory; and to
  store different irreducible representations (for SO(3) equivariant atomsitic
  features)

- Most of the calculators can compute gradients with respect to `positions`,
  `cell` or `stress`, storing them in the `gradient()` of metatensor's
  `TensorBlock`.

- All the native calculators are exposed through a C API, and accessible from
  multiple languages: Rust, C++ and Python.

- Interface to mutliple system providers, and a way to define custom system
  providers in user code. The following system providers are supported from
  Python: ASE (https://wiki.fysik.dtu.dk/ase/); chemfiles
  (https://chemfiles.org/); and PySCF (https://pyscf.org/)

- Python-only calculators, based on Clebsch-Gordan tensor products to combine
  equivariant featurizations. This includes
    - PowerSpectrum, able to combine two different spherical expansions
    - `EquivariantPowerSpectrum`, the same but producing features both invariant
      and covariant with respect to rotations
    - `DensityCorrelations` to compute arbitrary body-order density correlations;
    - `ClebschGordanProduct`, the core building block that does a single
      Clebsch-Gordan tensor product.

- Python tools to define custom atomic density and radial basis functions, and
  then compute splines for the radial integral apearing in SOAP and LODE
  spherical expansions. This enables using these native calculators with
  user-defined atomic densities and basis functions.
