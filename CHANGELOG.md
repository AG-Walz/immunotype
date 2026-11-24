# immunotype: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.6]

### Changed
- Changed the code license from MIT to AGPL-3.0-or-later, following the [Open Science Policy of the University of Tübingen](https://uni-tuebingen.de/universitaet/profil/werte-und-visionen/open-science-an-der-universitaet-tuebingen/open-science-policy/) ([#23](https://github.com/AG-Walz/immunotype/pull/23))
- Licensed data files and model weights under CC BY 4.0 (`LICENSE-DATA`) ([#23](https://github.com/AG-Walz/immunotype/pull/23))
- Added Oliver Kohlbacher and Juliane Walz as authors ([#23](https://github.com/AG-Walz/immunotype/pull/23))
- Updated GitHub Actions to Node.js 24 versions ([#22](https://github.com/AG-Walz/immunotype/pull/22))

## [1.0.5]

### Fixed
- Gradio app ran out of memory on HF Space for large inputs; lowered max peptides per batch default to 10,000 ([#21](https://github.com/AG-Walz/immunotype/pull/21))

## [1.0.4]

### Changed
- Added bioRxiv preprint citation in README and Gradio app help tab ([doi:10.64898/2026.04.08.717021](https://doi.org/10.64898/2026.04.08.717021)) ([#18](https://github.com/AG-Walz/immunotype/pull/18))

### Fixed
- Gradio app showed both light and dark logos at once when not launched via the CLI (e.g. on HF Space) ([#19](https://github.com/AG-Walz/immunotype/pull/19))

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
