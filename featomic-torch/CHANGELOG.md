# Changelog

All notable changes to featomic are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/featomic/)

### Added

- Optional `dtype` in `systems_to_torch` to control positions and cell precision.

<!-- Possible sections for each package:

### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.7.6](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.6) - 2026-10-01

### Added

- Added support for torch v2.13 and v2.14

## [Version 0.7.5](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.5) - 2026-06-25

### Changed

- The code is now compatible with metatensor-torch v0.10

## [Version 0.7.4](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.4) - 2026-05-15

### Changed

- Added support for torch v2.11 and v2.12
- The code is now compatible with metatensor-torch v0.9.0

## [Version 0.7.3](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.3) - 2026-02-03

### Changed

- Added support for torch v2.10

## [Version 0.7.2](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.2) - 2025-12-04

- We now require Python >= 3.10
- Added support for torch v2.9, removed support for torch 2.1 and 2.2

## [Version 0.7.1](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.1) - 2025-09-17

- Added support for torch v2.8, and metatensor-torch v0.8.0

## [Version 0.7.0](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.7.0) - 2025-05-22

- featomic-torch now integrates with [metatomic-torch](https://github.com/metatensor/metatomic)
  instead of metatensor-torch atomistic models.

## [Version 0.6.2](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.6.2) - 2025-05-21

- Add Support for PyTorch 2.7, removed support for PyTorch 2.0 and older.

## [Version 0.6.1](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.6.1) - 2025-02-18

- Add Support for Python 3.13 and PyTorch 2.6


## [Version 0.6.0](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.6.0) - 2025-01-07

### Added

- C++ and Python TorchScript bindings to `featomic`, making all calculators
  accessible from TorchScript models.

- Integration of the TorchScript calculators with
  [metatensor-torch-atomistic](https://docs.metatensor.org/latest/atomistic/index.html),
  using the `System` class from this package as a system provider, and
  integrating with neighbor lists provided by the simulation engine through
  metatensor.

- Automatic integration with (Py)Torch automatic differentiation system. If any
  of the inputs requires gradients, then `featomic-torch` will compute them,
  store them, and integrate them with a custom backward function on the
  calculator output.

- Re-export of Python tools for Clebsch-Gordan tensor products from `featomic`,
  in a TorchScript-compatible way.
