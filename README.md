# Hybrid storage sizing for data center workloads

Exploratory analysis of AI data center workload power profiles for sizing a
battery and supercapacitor storage system using [SimSES](https://github.com/tum-ees/simses).
The goal of this project is exploring the size and efficiency trade-offs of hybrid systems.

- [x] **Exploring the power profiles**
  ([notebooks/eda.ipynb](notebooks/eda.ipynb)).
  Calculated power/energy demands for further analyses.
  Identified stitch artefacts -> individual trials were used in subsequent analyses.

- [x] **Implement a hybrid storage controller**
  ([src/hess.py](src/hess.py)).

  Implemented hybrid controller uses a low-pass filter to divide power requests.
  The battery string handles the slower component, the supercapacitor string covers the remainder.

- [x] **Estimate storage requirements with a rough static sizing analysis**
  ([notebooks/static.ipynb](notebooks/static.ipynb)).

  Calculated minimal storage requirements to satisfy each trials power and energy demands.
  Ran dynamic simulation checks to confirm designs plausibility.

- [ ] **Optimise storage size, energy losses, and filter parameters across trials for different scenarios**
  ([notebooks/sim2.ipynb](notebooks/sim2.ipynb)).

  Implemented parameter sweep exploring battery and supercapacitor configuration
  as well as filter time constants on one trial.
  TODO:
  - extend simulation across all trials
  - current "scenario" simulates the storage system being the sole provider of electricity;
    add additional scenarios with power grid and different constraints,
    such as minimal post-trail remaining SOC

Source data: [High-resolution AI Data Center Training Workloads Dataset](https://figshare.com/articles/dataset/High-resolution-AI-Data-Center-Training-Workloads-Dataset/31654879?file=62953543).
The notebooks expect local files in `data/` at the repository root, 
including `stitched_profiles.csv` and per-profile files.

The notebooks use Python 3.13+; dependencies are recorded in
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).
