# pyrao
`pyrao` is a few things:
 - a Python wrapper for [`rao`](https://github.com/jcranney/rao) package - a set
   of [Adaptive Optics](https://en.wikipedia.org/wiki/Adaptive_optics) (AO) 
   tools written in Rust,
 - a standalone AO simulator with Python APIs, fast enough to run 8m class 
   simulations at real time on a modest laptop,
 - a data stream generator for developing tools based on [ImageStreamIO](
   https://github.com/milk-org/ImageStreamIO),
 - (todo) a [Gymnasium](https://gymnasium.farama.org/) formatted [environment](
   https://gymnasium.farama.org/environments/third_party_environments/) 
   for developping and testing reinforcement learning.
 - an experiment in linear algebra + statistics, optimal control/estimation, 
   python-wrapped-rust (using [PyO3](https://github.com/PyO3/pyo3)).
 - (todo) a performance evaluation tool - provided you can simulate your system
   in `rao`.

There are many things that `pyrao` *is not*, but most importantly:
 - `pyrao` is not an "end-to-end numerical simulation tool for AO" (see 
   [#assumptions])
 - `pyrao` is not an RTC in its own right, though it emulates some
   functionalities of one.

### Assumptions
We assume that everything in the AO loop is linear, and all sources of noise
are additive Gaussian *iid* processes. For example, we assume that the
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