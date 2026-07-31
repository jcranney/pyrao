![PyPI - Version](https://img.shields.io/pypi/v/rao)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/jcranney/pyrao/CI.yml)

# pyrao

`pyrao` is a few things:

- a Python wrapper for [`rao`](https://github.com/jcranney/rao) package - a set
  of [Adaptive Optics](https://en.wikipedia.org/wiki/Adaptive_optics) (AO)
  tools written in Rust,
- a standalone AO simulator with Python APIs, fast enough to run 8m class
  simulations at real time on a modest laptop,
- a data stream generator for developing tools based on [ImageStreamIO](https://github.com/milk-org/ImageStreamIO),
- an experiment in linear algebra + statistics, optimal control/estimation,
  python-wrapped-rust (using [PyO3](https://github.com/PyO3/pyo3)).

There are many things that `pyrao` _is not_, but most importantly:

- `pyrao` is not an "end-to-end numerical simulation tool for AO" (see
  [#assumptions])
- `pyrao` is not an RTC in its own right, though it emulates some
  functionalities of one.

`pyrao` is also suitable for the following tasks, but has not yet been developed for them:

- a [Gymnasium](https://gymnasium.farama.org/) formatted [environment](https://gymnasium.farama.org/environments/third_party_environments/)
  for developping and testing reinforcement learning.
- a performance evaluation tool - provided you can simulate your system
  in `rao`.

If there are tasks you think `pyrao` could be suitable for and you would like to see them developed, [raise an issue](https://github.com/jcranney/pyrao/issues).

### Installation

Annoyingly, there is already a PyPI package named `pyrao`, so to install with
pip, you should use:

```bash
pip install rao
```

but then to import the package, use (as expected):

```python
import pyrao
```

### Usage

The usage of this wrapper is very actively changing, based on my own needs.
Currently, the main use-case for the Python wrapper is for rapid generation of
interaction matrices and covariance matrices, both for linear simulations of AO
systems, and for fitting of parameters by comparing measured and analytical
matrices.

For example usages, see the following:

- https://github.com/jcranney/mavis-saturations, a simulation for investigating the
  effects of saturation and NCPAs in the MAVIS control scheme.
- https://github.com/jcranney/mavis-misreg, a simulated validation for the fitting
  of system parameters using a synthetic interaction matrix.

### Assumptions

We assume that everything in the AO loop is linear, and all sources of noise
are additive Gaussian _iid_ processes. For example, we assume that the
measurements are a linear combination of atmospheric phase (according to some
sampling of von Karman layers), actuator commands (according to some influence
functions), and a noise vector with a specified covariance matrix.

### Disclaimer

This is presently a hobby-project, so development may be slow and/or
unpredictable. However, if you have a change you would like to see, or a
feature you would like added, I encourage you to file an issue - since it's
likely something I haven't considered and it could prove useful to others. If
you make a change yourself and you think others might also find it useful,
please consider making a pull request so that I can include your edits in this
repo. If you have any other feedback, feel free to share it with me directly
via email: [jesse.cranney@anu.edu.au](mailto:jesse.cranney@anu.edu.au).

### Refactoring Effort

Say I wanted to refactor the current API, namely to allow the python user to specify
the following parameters through intuitive calls:

```rust
teldiam: f64,               // diameter of telescope in metres
cobs: f64,                  // central obscuration, fraction of diameter
coupling: f64,              // coupling between DM actuators
nactux: u32,                // number of actuators across DM diameter
dmalt: f64,                 // dm altitude in metres
pitch: f64,                 // dm pitch in metres
nsubx: u32,                 // number of subapertures across WFS pupil
ntssamples: u32,            // number of samples across TS pupil
nphisamples: u32,           // number of phase samples across pupil,
wfs_dirs: Vec<(f64, f64)>,  // directions of WFSs (arcsec)
ts_dirs: Vec<(f64, f64)>,   // directions of WFSs (arcsec)
dm_delta: (f64, f64),       // dm position offset (metres)
wfs_delta: Vec<(f64, f64)>, // wfs position offset (metres)
dm_clocking: f64,           // dm rotation (radians)
wfs_clocking: Vec<f64>,     // wfs rotation (radians)
dm_zoom: f64,               // dm magnification error (0.0 === unity magnification)
wfs_zoom: Vec<f64>,         // wfs magnification error (0.0 === unity magnification)
gsalt: f64,                 // guide star altitude
microns_per_volt: f64,      // microns per volt of actuators
cn2: cn2_profile_tbd,       // cn2 profile, tbd
```

First pass at recollecting these:

```rust
/// The scope of a system is precisely enough to uniquely generate the matrices
/// required for a single AO loop, or to simulate an AO system.
struct System {
  telescope: Telescope,
  dm: Vec<Dm>,
  wfs: Vec<Wfs>,
  ctrl: Ctrl,
  atmos: Atmos,
}

/// Physical telescope parameters that may be cloned between different AO systems
struct Telescope {
  teldiam: f64,               // diameter of telescope [m]
  cobs: f64,                  // central obscuration [m]
}

/// A single deformable mirror. A collection of actuators.
struct Dm {
  alt: Altitude,                   // dm altitude [m]
  coupling: f64,              // coupling between DM actuators
  microns_per_volt: f64,      // microns per volt of actuators
  actu_pos: Positions         // actuator positions in plane [m]
  misreg: MisReg,             // misregistration parameters
}

/// A single wavefront sensor. A collection of measurements.
struct Wfs {
  dir: (f64, f64),            // directions of WFS [rad]
  gsalt: Altitude,                 // guide star altitude [m]
  subap_pos: Positions        // positions of centre of subapertures in pupil [m]
  misreg: MisReg,             // misregistration parameters
}

/// Misregistration from the nominal object geometry. Common for WFSs and DMs.
struct MisReg {
  delta: (f64, f64),          // position offset [m]
  clocking: f64,              // rotation [rad]
  zoom: f64,                  // magnification error (1.0 => unity magnification)
}

/// Common positions 
enum Positions {
  RectGrid {
    width: usize,
    negative_corner: (f64, f64),
    positive_corner: (f64, f64),
  },
  Explicit(Vec<(f64,f64)>),
}

/// Control configuration
struct Ctrl {
  nsamples: u32,              // number of samples across TS pupil
  opt_dirs: Vec<(f64, f64)>,  // directions of WFSs [rad]
  nphisamples: u32,           // number of phase samples across pupil,
}

/// A layer of von karman turbulence
struct TurbLayer {
  r0: f64,                // r0 of this layer [m]
  outer_scale: f64,
  alt: Altitude,          // altitude of layer [m]
  windx: f64,             // wind speed (x-component) [m/s]
  windy: f64,             // wind speed (y-component) [m/s]
}

/// Atmosphere
enum Atmos {
  VonKarman(Vec<TurbLayer>),
}

enum Altitude {
  Finite(f64),
  Infinite,
}
```
