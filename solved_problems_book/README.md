# Solved Problems in Geostatistics

Python implementations of problems from the textbook *"Solved Problems in Geostatistics"*
by Oy Leuangthong, K. Daniel Khan, and Clayton V. Deutsch (Wiley, 2008, ISBN 978-0-470-17792-1).

## Problem Index

| Chapter | Problem | Directory | Description |
|---------|---------|-----------|-------------|
| 2.3 | Standardization and Probability Intervals | `2.3.Standartization and Probability Intervals/` | Normal score transform and probability interval computation |
| 3.1 | Basic Declustering | `3.1. Basic Declustering/` | Cell declustering for irregular sampling |
| 3.3 | Comparison of Declustering Methods | `3.3.Comparison of Declustering Methods/` | Cell vs polygonal declustering comparison |
| 4.1 | Central Limit Theorem | `4.1. Central limit theorem/` | CLT demonstration with spatial data |
| 4.2 | Bootstrap and Spatial Bootstrap | `4.2.Bootstrap and Spatial Bootstrap/` | Bootstrap confidence intervals, spatial bootstrap |
| 4.3 | Transfer of Uncertainty (NPV) | `4.3. Transfer of Uncertainty/` | Uncertainty propagation through NPV calculation |
| 5.2 | Variogram Calculation | `5.2. Variogram Calculation/` | 2D and 3D experimental variogram computation |
| 5.3 | Variogram Modeling and Volume Variance | `5.3. Variogram Modeling and Volume Variance/` | Variogram model fitting and volume-variance relations |
| 7.2 | Conditioning by Kriging | `7.2. Conditioning by Kriging/` | Kriging for conditional simulation |
| 7.3 | Gaussian Simulation | `7.3. Gaussian Simulation/` | Sequential Gaussian Simulation (SGS) |
| 8.3 | Indicator Simulation for Categorical Data | `8.3. Indicator Simulation for Categorical Data/` | Sequential Indicator Simulation (SIS) |

## Shared Modules (`shared/`)

| Module | Description |
|--------|-------------|
| `statistics.py` | Statistical functions |
| `gslib.py` | GSLIB file read/write |
| `gaussian_cdf.py` | Gaussian CDF computation |
| `grid_3d.py` | 3D grid class |
| `geo_routines.py` | Geostatistics routines |
| `variogram_routines.py` | Variogram routines |
| `opt_x_y.py` | 2D data classes |
| `decl_grid.py` | Declustering grid |

## Test Comparison (`Examples test_compare/`)

| Script | Description |
|--------|-------------|
| `indicator_kriging.py` | IK test comparison |
| `make_prop.py` | Property generation |
| `sis.py` | SIS test comparison |

**Note:** These scripts are educational implementations that may differ from the
HPGL library API. They were written before HPGL Reborn and use their own utility
modules rather than the modern `geo_bsd` package.

See the main [README.md](../readme.md) for HPGL library documentation.
