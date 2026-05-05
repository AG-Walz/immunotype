# immunotype: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Added bioRxiv preprint citation in README and Gradio app help tab ([doi:10.64898/2026.04.08.717021](https://doi.org/10.64898/2026.04.08.717021))

## [1.0.3]

### Fixed
- Gradio app probability output kept running and could not be accessed

## [1.0.2]

### Changed
- Bump gradio dependency to `>=6.0.0`, use `buttons` API for Dataframe copy button

## [1.0.1]

### Fixed
- Added `matplotlib` to app dependencies (required by pandas `background_gradient`)

## [1.0.0]

- GNN + lookup table ensemble prediction for HLA class I allotyping
- CLI with `immunotype` command
- Gradio web interface with dark theme and light/dark mode logo support
- Python API via `from immunotype import predict`
- PyPI publishing via GitHub Actions with OIDC trusted publishing
- Separate HF Space deployment using `immunotype[app]` from PyPI
