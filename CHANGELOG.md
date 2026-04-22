# Changelog

All notable changes to PyroPT are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
PyroPT uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-04-22

### Added
- Representative ±1σ error bar on P-T plot, dynamically positioned in the upper right of the plot area
- Sidebar checkbox ("Show Error Bars") to toggle error bar visibility, defaulting to on
- Warning suppression is now conditional: warnings are visible when running locally with `--local-mode`, and suppressed on Streamlit Cloud for a clean user experience with the online app

### Changed
- **KNN model updated** from default parameters to `weights='distance'` and `p=1` (Manhattan distance), giving closer training samples greater influence on predictions
- **Pressure model feature set expanded** from 4 to 6 oxides: Al₂O₃, Cr₂O₃, TiO₂, MgO, CaO, SiO₂
- **Temperature model feature set** expanded from 4 to 6 oxides: MnO, FeO, TiO₂, MgO, CaO, SiO₂
- **Training data** — extreme samples (top and bottom 12 per garnet type by P or T) are now duplicated in the training set to upweight boundary values and reduce regression-to-the-mean at high and low P-T
- **Random seeds updated** based on testing values and choosing a median value from the performance distribution: pressure model 141, temperature model 247
- Footer citation now derives version number and year dynamically from `PYROPT_VERSION`, so it updates automatically with future releases
- Footer DOI updated to the general Zenodo archive DOI (10.5281/zenodo.17400965)
- `citation.cff` version updated to 1.2.0
- `.gitignore` replaced with comprehensive GitHub Python template
- Minor related updates in `README.md` text

### Fixed
- **Fatal crashes in the app associated with updates to how pandas handled some values have been addressed.** Made other updates related to prior reliance on deprecated features / behaviours.
- Fixed where `float()` called on a single-element pandas Series to now correctly extract the scalar via `.iloc[0]`, resolving a `TypeError` under pandas 3.0+
- Neighbour string columns are now pre-initialised as `object` dtype before assignment loop, resolving a `TypeError` when pandas 3.0+ refused to upcast `float64` columns to accept string values
- Filtered training DataFrame now explicitly copied (`.copy()`) before column assignment, resolving a `SettingWithCopyWarning` under pandas Copy-on-Write
- `groupby().apply()` updated with `include_groups=False` for forward compatibility with pandas 3.0+

---

## [1.0.1] - 2025-10-20

### Added
- **Initial public release of PyroPT**
- KNN-based pressure and temperature prediction from individual Cr-rich peridotitic pyrope garnet
- Fe³⁺ recalculation using the Droop (1987) method
- Garnet classification (G9, G10, G11) with validation against training data ranges
- Geotherm fitting with weighted linear regression and LAB depth estimation
- Streamlit web interface with sidebar controls for plot customisation
- Support for local execution with `--local-mode` flag for relaxed upload limits
- Downloadable predictions as CSV and plot as PDF
