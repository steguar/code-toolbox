# Community-Based Epidemic and Opinion Dynamics Platform

This repository contains the computational framework developed within
the project for studying:

-   Community-structured epidemic diffusion
-   Media and awareness effects
-   Ideological heterogeneity
-   Opinion dynamics
-   Synthetic and randomized network generation

------------------------------------------------------------------------

## Repository Structure

    .
    ├── epidemic/
    │   ├── multi_type_epidemic.py
    │   ├── config.json
    │   └── input/
    │
    ├── opinion/
    │   ├── opinion_dynamics.py
    │   ├── config.json
    │   └── input/
    │
    ├── network/
    │   ├── config.json
    │   └── geometric_block_model/
    │       └── src/rhbm/
    │           ├── rhbm_generate.py
    │           └── rhbm_randomize.py
    │
    ├── gui/
    │   ├── epidemic_gui.py
    │   ├── opinion_gui.py
    │   └── network_gui.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 1. Epidemic Model

Implements a multi-type extension of SIR/SIS with:

-   Community-dependent mixing
-   Media-intensity feedback
-   Optional exposed stage (SEIR/SEIS)
-   Mean-field and network simulation modes

Run:

``` bash
cd epidemic
python multi_type_epidemic.py
```

GUI:

``` bash
cd gui
python epidemic_gui.py
```

------------------------------------------------------------------------

## 2. Opinion Dynamics Model

Implements:

-   Community-based opinion evolution
-   Social interaction term
-   External risk-driven modulation
-   Radicalization vs prudence regimes

Run:

``` bash
cd opinion
python opinion_dynamics.py
```

GUI:

``` bash
cd gui
python opinion_gui.py
```

------------------------------------------------------------------------

## 3. Network Generation

Two tools are available:

-   `rhbm_generate.py`: synthetic network generation
-   `rhbm_randomize.py`: randomization of observed networks

Located in:

    network/geometric_block_model/src/rhbm/

GUI:

``` bash
cd gui
python network_gui.py
```

------------------------------------------------------------------------

## Installation

Clone repository:

``` bash
git clone https://gitlab.com/YOUR_GROUP/YOUR_REPO.git
cd YOUR_REPO
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Reproducibility

All simulations are parameter-driven via JSON configuration files.

Random seeds are controlled in configuration files.

------------------------------------------------------------------------

## License

MIT License.

------------------------------------------------------------------------

## Citation

If you use this software in academic work, please cite:

> \[Insert project citation or DOI here\]
