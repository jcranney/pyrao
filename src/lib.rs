use std::fs;

use pyo3::prelude::*;
use rao::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

const AS2RAD: f64 = 4.848e-6;

/// A Python module implemented in Rust.
#[pymodule]
pub mod pyrao {
    use super::*;

    #[derive(Debug, Clone)]
    struct VonKarmanLayers {
        layers: Vec<VonKarmanLayer>,
    }

    impl CoSampleable for VonKarmanLayers {
        fn cosample(&self, p: &Line, q: &Line, dt: f64) -> f64 {
            self.layers
                .iter()
                .map(|layer| layer.cosample(p, q, dt))
                .sum()
        }
    }

    /// The scope of a system is precisely enough to uniquely generate the matrices
    /// required for a single AO loop, or to simulate an AO system.
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct System {
        telescope: Telescope,
        dm: Vec<Dm>,
        wfs: Vec<Wfs>,
        ctrl: Ctrl,
        atmos: Atmos,
    }
    #[pymethods]
    impl System {
        #[staticmethod]
        fn new(telescope: Telescope, dm: Vec<Dm>, wfs: Vec<Wfs>, ctrl: Ctrl, atmos: Atmos) -> Self {
            Self {
                telescope,
                dm,
                wfs,
                ctrl,
                atmos,
            }
        }
    }

    /// Physical telescope parameters that may be cloned between different AO systems
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct Telescope {
        teldiam: f64, // diameter of telescope [m]
        cobs: f64,    // central obscuration [m]
    }
    #[pymethods]
    impl Telescope {
        #[staticmethod]
        fn new(teldiam: f64, cobs: f64) -> Self {
            Self { teldiam, cobs }
        }
    }

    /// A single deformable mirror. A collection of actuators.
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct Dm {
        alt: Altitude,         // dm altitude [m]
        coupling: f64,         // coupling between DM actuators
        microns_per_volt: f64, // microns per volt of actuators
        actu_pos: Positions,   // actuator positions in plane [m]
        misreg: MisReg,        // misregistration parameters
    }
    #[pymethods]
    impl Dm {
        #[staticmethod]
        fn new(
            alt: Altitude,
            coupling: f64,
            microns_per_volt: f64,
            actu_pos: Positions,
            misreg: MisReg,
        ) -> Self {
            Self {
                alt,
                coupling,
                microns_per_volt,
                actu_pos,
                misreg,
            }
        }
    }

    impl From<&Dm> for Vec<Actuator> {
        fn from(value: &Dm) -> Self {
            let Dm {
                alt,
                coupling,
                microns_per_volt,
                actu_pos,
                misreg,
            } = value;
            let MisReg {
                delta,
                clocking,
                zoom,
            } = misreg;
            let com_coords: Vec<Vec2D> = actu_pos.into();
            com_coords
                .into_iter()
                .map(move |p| {
                    let x: f64 = (p.x * clocking.cos() + p.y * clocking.sin()) * zoom + delta.0;
                    let y: f64 = (-p.x * clocking.sin() + p.y * clocking.cos()) * zoom + delta.1;
                    rao::Actuator::Gaussian {
                        position: Vec3D::new(x, y, alt.into()),
                        sigma: rao::coupling_to_sigma(*coupling, actu_pos.pitch().0),
                        microns_per_volt: *microns_per_volt,
                    }
                })
                .collect()
        }
    }

    /// A single wavefront sensor. A collection of measurements.
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct Wfs {
        /// directions of WFS (radians)
        dir: Vec2D,
        /// guide star altitude (metres)
        gsalt: Altitude,
        /// positions of centre of subapertures in pupil (metres)
        subap_pos: Positions,
        /// misregistration parameters
        misreg: MisReg,
    }
    #[pymethods]
    impl Wfs {
        #[staticmethod]
        fn new(dir: (f64, f64), gsalt: Altitude, subap_pos: Positions, misreg: MisReg) -> Self {
            Self {
                dir: Vec2D { x: dir.0, y: dir.0 },
                gsalt,
                subap_pos,
                misreg,
            }
        }
    }

    impl From<&Wfs> for Vec<Measurement> {
        fn from(value: &Wfs) -> Self {
            let Wfs {
                dir,
                gsalt,
                subap_pos,
                misreg,
            } = value;
            let (dx, dy) = subap_pos.pitch();
            let centres: Vec<Vec2D> = subap_pos.into();
            let MisReg {
                delta,
                clocking,
                zoom,
            } = misreg;
            let mut meas: Vec<rao::Measurement> = vec![];
            // first the y-slopes ...
            meas.append(
                &mut centres
                    .iter()
                    .map(|p| {
                        let x0: f64 =
                            (p.x * clocking.cos() + p.y * clocking.sin()) * zoom + delta.0;
                        let y0: f64 =
                            (-p.x * clocking.sin() + p.y * clocking.cos()) * zoom + delta.1;
                        let l = Line::new(x0, dir.x, y0, dir.y);
                        rao::Measurement::SlopeTwoEdge {
                            central_line: l.clone(),
                            edge_length: dx,
                            edge_separation: dy,
                            gradient_axis: Vec2D::new(clocking.sin(), clocking.cos()),
                            npoints: 1,
                            altitude: gsalt.into(),
                        }
                    })
                    .collect::<Vec<rao::Measurement>>(),
            );
            // ... then the x-slopes
            meas.append(
                &mut centres
                    .iter()
                    .map(|p| {
                        let x0: f64 =
                            (p.x * clocking.cos() + p.y * clocking.sin()) * zoom + delta.0;
                        let y0: f64 =
                            (-p.x * clocking.sin() + p.y * clocking.cos()) * zoom + delta.1;
                        let l = Line::new(x0, dir.x, y0, dir.y);
                        rao::Measurement::SlopeTwoEdge {
                            central_line: l.clone(),
                            edge_length: dy,
                            edge_separation: dx,
                            gradient_axis: Vec2D::new(clocking.cos(), -clocking.sin()),
                            npoints: 1,
                            altitude: gsalt.into(),
                        }
                    })
                    .collect::<Vec<rao::Measurement>>(),
            );
            meas
        }
    }

    impl From<&Wfs> for Vec<Line> {
        fn from(value: &Wfs) -> Self {
            let meas: Vec<Measurement> = value.into();
            meas.iter()
                .map(|m| match m {
                    rao::Measurement::Zero => Line::new_on_axis(0.0, 0.0),
                    rao::Measurement::Phase { line } => line.clone(),
                    rao::Measurement::SlopeTwoLine { .. } => todo!(),
                    rao::Measurement::SlopeTwoEdge { central_line, .. } => central_line.clone(),
                })
                .collect()
        }
    }

    /// Misregistration from the nominal object geometry. Common for WFSs and DMs.
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct MisReg {
        delta: (f64, f64), // position offset [m]
        clocking: f64,     // rotation [rad]
        zoom: f64,         // magnification error (1.0 => unity magnification)
    }
    #[pymethods]
    impl MisReg {
        #[staticmethod]
        fn new(delta: (f64, f64), clocking: f64, zoom: f64) -> Self {
            Self {
                delta,
                clocking,
                zoom,
            }
        }
    }

    /// Common positions
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    enum Positions {
        RectGrid {
            width: usize,
            negative_corner: (f64, f64),
            positive_corner: (f64, f64),
        },
        Explicit {
            pos: Vec<(f64, f64)>,
            pitch: (f64, f64),
        },
    }
    #[pymethods]
    impl Positions {
        #[staticmethod]
        fn rect_grid(
            width: usize,
            negative_corner: (f64, f64),
            positive_corner: (f64, f64),
        ) -> Self {
            Self::RectGrid {
                width,
                negative_corner,
                positive_corner,
            }
        }
    }

    impl Positions {
        pub fn pitch(&self) -> (f64, f64) {
            match self {
                Positions::RectGrid {
                    width,
                    negative_corner,
                    positive_corner,
                } => (
                    (positive_corner.0 - negative_corner.0) / *width as f64,
                    (positive_corner.1 - negative_corner.1) / *width as f64,
                ),
                Positions::Explicit { pitch, .. } => *pitch,
            }
        }
    }

    impl From<&Positions> for Vec<Vec2D> {
        fn from(value: &Positions) -> Self {
            match value {
                Positions::RectGrid {
                    width,
                    negative_corner,
                    positive_corner,
                } => {
                    let x = Vec2D::linspace(
                        &Vec2D::new(negative_corner.0, 0.0),
                        &Vec2D::new(positive_corner.0, 0.0),
                        *width as u32,
                    );
                    let y = Vec2D::linspace(
                        &Vec2D::new(0.0, negative_corner.0),
                        &Vec2D::new(0.0, positive_corner.0),
                        *width as u32,
                    );
                    x.into_iter().zip(y).map(|(x, y)| x + y).collect()
                }
                Positions::Explicit { pos, .. } => {
                    pos.into_iter().map(|(x, y)| Vec2D::new(*x, *y)).collect()
                }
            }
        }
    }

    /// Control configuration
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct Ctrl {
        /// directions of truth-sensors (radians)
        opt_dirs: Vec<(f64, f64)>,
        /// positions of ts samples in pupil (metres)
        pos: Positions,
        /// time delay used for covariance computation
        dt: f64,
    }
    #[pymethods]
    impl Ctrl {
        #[staticmethod]
        fn new(opt_dirs: Vec<(f64, f64)>, pos: Positions, dt: f64) -> Self {
            Self { opt_dirs, pos, dt }
        }
    }
    impl From<&Ctrl> for Vec<Measurement> {
        fn from(value: &Ctrl) -> Self {
            let Ctrl { opt_dirs, pos, .. } = value;
            let points: Vec<Vec2D> = pos.into();
            opt_dirs
                .iter()
                .flat_map(|dir| {
                    points
                        .clone()
                        .into_iter()
                        .map(|p| Measurement::Phase {
                            line: Line::new(p.x, dir.0, p.y, dir.1),
                        })
                        .collect::<Vec<Measurement>>()
                })
                .collect()
        }
    }

    /// A layer of von karman turbulence
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct TurbLayer {
        r0: f64, // r0 of this layer [m]
        outer_scale: f64,
        alt: Altitude, // altitude of layer [m]
        windx: f64,    // wind speed (x-component) [m/s]
        windy: f64,    // wind speed (y-component) [m/s]
    }
    #[pymethods]
    impl TurbLayer {
        #[staticmethod]
        fn new(r0: f64, outer_scale: f64, alt: Altitude, windx: f64, windy: f64) -> Self {
            Self {
                r0,
                outer_scale,
                alt,
                windx,
                windy,
            }
        }
    }
    impl From<&TurbLayer> for VonKarmanLayer {
        fn from(value: &TurbLayer) -> Self {
            let TurbLayer {
                r0,
                outer_scale,
                alt,
                windx,
                windy,
            } = value;
            Self::new(
                *r0,
                *outer_scale,
                alt.into(),
                Vec2D {
                    x: *windx,
                    y: *windy,
                },
            )
        }
    }

    /// Atmosphere
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    enum Atmos {
        VonKarman(Vec<TurbLayer>),
    }
    #[pymethods]
    impl Atmos {
        #[staticmethod]
        fn new(layers: Vec<TurbLayer>) -> Self {
            Self::VonKarman(layers)
        }
    }
    impl From<&Atmos> for Vec<VonKarmanLayer> {
        fn from(value: &Atmos) -> Self {
            match value {
                Atmos::VonKarman(turb_layers) => {
                    turb_layers.iter().map(|layer| layer.into()).collect()
                }
            }
        }
    }

    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    enum Altitude {
        Finite(f64),
        Infinite(),
    }
    #[pymethods]
    impl Altitude {
        #[staticmethod]
        fn new(h: f64) -> Self {
            if h.is_finite() {
                Self::Finite(h)
            } else {
                Self::Infinite()
            }
        }
    }

    impl From<&Altitude> for f64 {
        fn from(value: &Altitude) -> Self {
            match value {
                Altitude::Finite(x) => *x,
                Altitude::Infinite() => f64::INFINITY,
            }
        }
    }

    impl From<&System> for SystemGeom {
        fn from(value: &System) -> Self {
            let System {
                telescope,
                dm,
                wfs,
                ctrl,
                atmos,
            } = value;
            let meas = wfs
                .iter()
                .flat_map::<Vec<Measurement>, _>(|w| w.into())
                .collect();
            let meas_lines = wfs.iter().flat_map::<Vec<Line>, _>(|w| w.into()).collect();
            let com = dm
                .iter()
                .flat_map::<Vec<Actuator>, _>(|d| d.into())
                .collect();
            let Telescope { teldiam, cobs } = telescope;
            let pupil = Pupil {
                rad_outer: *teldiam / 2.0,
                rad_inner: cobs * teldiam / 2.0,
                spider_thickness: 0.0,
                spiders: vec![],
            };
            Self {
                meas,
                phi: vec![],
                ts: ctrl.into(),
                com,
                cov_model: atmos.into(),
                pupil: Some(pupil),
                meas_lines,
                simul_dt: 0.0,
                meas_dt: ctrl.dt,
            }
        }
    }

    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct SystemGeom {
        meas: Vec<Measurement>,         // measurements
        phi: Vec<Measurement>,          // latent phase, for explicit reconstruction
        ts: Vec<Measurement>,           // truth-sensors, for "laa"-based reconstruction
        com: Vec<Actuator>,             // commands
        cov_model: Vec<VonKarmanLayer>, // turbulence covariance
        pupil: Option<Pupil>,           // optional pupil function
        meas_lines: Vec<Line>, // lines in direction of measurements, used for pupil evaluation
        simul_dt: f64,         // dt, for latent phase covariance calculations
        meas_dt: f64,          // dt, for meas-ts covariance calculations
    }

    impl SystemDefinition for SystemGeom {
        fn com(&self) -> &Vec<Actuator> {
            &self.com
        }

        fn com_mut(&mut self) -> &mut Vec<Actuator> {
            &mut self.com
        }

        fn meas(&self) -> &Vec<Measurement> {
            &self.meas
        }

        fn meas_mut(&mut self) -> &mut Vec<Measurement> {
            &mut self.meas
        }

        fn meas_lines(&self) -> &Vec<Line> {
            &self.meas_lines
        }

        fn meas_lines_mut(&mut self) -> &mut Vec<Line> {
            &mut self.meas_lines
        }

        fn pupil(&self) -> Option<&Pupil> {
            self.pupil.as_ref()
        }
    }

    #[pymethods]
    impl SystemGeom {
        #[staticmethod]
        fn new_empty() -> Self {
            Self {
                meas: vec![],
                phi: vec![],
                ts: vec![],
                com: vec![],
                cov_model: vec![],
                pupil: None,
                meas_lines: vec![],
                simul_dt: 0.0,
                meas_dt: 0.0,
            }
        }

        fn save_yaml(&self, filename: &str) {
            SystemDefinition::save_yaml(self, filename);
        }

        #[staticmethod]
        fn load_yaml(filename: &str) -> Self {
            SystemDefinition::load_yaml(filename)
        }

        fn set_dt(&mut self, simul_dt: f64, meas_dt: f64) {
            self.simul_dt = simul_dt;
            self.meas_dt = meas_dt;
        }

        fn add_phi(&mut self, teldiam: f64, nphisamples: u32) {
            /////////////
            // define phi related coordinates:
            let xx = Vec2D::linspread(
                &Vec2D::new(-teldiam * 0.5, 0.0),
                &Vec2D::new(teldiam * 0.5, 0.0),
                nphisamples,
            );
            let yy = Vec2D::linspread(
                &Vec2D::new(0.0, -teldiam * 0.5),
                &Vec2D::new(0.0, teldiam * 0.5),
                nphisamples,
            );
            let phi_coords: Vec<Vec2D> = xx
                .iter()
                .flat_map(|x| yy.iter().map(move |y| x + y))
                .collect();

            let mut phi: Vec<rao::Measurement> = phi_coords
                .iter()
                .map(|p0| rao::Measurement::Phase {
                    line: Line::new_on_axis(p0.x, p0.y),
                })
                .collect();

            self.phi.append(&mut phi);
        }

        fn add_ts(&mut self, teldiam: f64, ntssamples: u32, ts_dirs: Vec<(f64, f64)>) {
            /////////////
            // define truth sensor related coordinates:
            let xx = Vec2D::linspace(
                &Vec2D::new(-teldiam * 0.5, 0.0),
                &Vec2D::new(teldiam * 0.5, 0.0),
                ntssamples,
            );
            let yy = Vec2D::linspace(
                &Vec2D::new(0.0, -teldiam * 0.5),
                &Vec2D::new(0.0, teldiam * 0.5),
                ntssamples,
            );
            let ts_coords: Vec<Vec2D> = xx
                .iter()
                .flat_map(|x| yy.iter().map(move |y| x + y))
                .collect();

            let mut ts: Vec<Measurement> = ts_dirs
                .into_iter()
                .map(|(x_as, y_as)| {
                    ts_coords.iter().map(move |p0| rao::Measurement::Phase {
                        line: Line::new(p0.x, x_as * AS2RAD, p0.y, y_as * AS2RAD),
                    })
                })
                .flatten()
                .collect();
            self.ts.append(&mut ts);
        }

        fn add_meas(
            &mut self,
            teldiam: f64,
            nsubx: u32,
            wfs_dirs: Vec<(f64, f64)>,
            gsalt: f64,
            wfs_delta: Vec<(f64, f64)>,
            wfs_clocking: Vec<f64>,
            wfs_zoom: Vec<f64>,
        ) {
            /////////////
            // define rao::Measurement related coordinates:
            let xx = Vec2D::linspread(
                &Vec2D::new(-teldiam * 0.5, 0.0),
                &Vec2D::new(teldiam * 0.5, 0.0),
                nsubx,
            );
            let yy = Vec2D::linspread(
                &Vec2D::new(0.0, -teldiam * 0.5),
                &Vec2D::new(0.0, teldiam * 0.5),
                nsubx,
            );
            let meas_coords: Vec<Vec2D> = xx
                .iter()
                .flat_map(|x| yy.iter().map(|y| x + y).collect::<Vec<Vec2D>>())
                .collect();
            let _wfs_dirs: Vec<Vec2D> = wfs_dirs
                .into_iter()
                .map(|(x, y)| Vec2D::new(x, y))
                .collect();

            let mut meas: Vec<rao::Measurement> = _wfs_dirs
                .iter()
                .enumerate()
                .map(|(dir_idx, dir_arcsec)| (dir_idx, dir_arcsec * AS2RAD))
                .flat_map(|(dir_idx, dir)| {
                    vec![
                        meas_coords
                            .iter()
                            .map(|p| {
                                let x0: f64 = (p.x * wfs_clocking[dir_idx].cos()
                                    + p.y * wfs_clocking[dir_idx].sin())
                                    * (1.0 + wfs_zoom[dir_idx])
                                    + wfs_delta[dir_idx].0;
                                let y0: f64 = (-p.x * wfs_clocking[dir_idx].sin()
                                    + p.y * wfs_clocking[dir_idx].cos())
                                    * (1.0 + wfs_zoom[dir_idx])
                                    + wfs_delta[dir_idx].1;
                                let l = Line::new(x0, dir.x, y0, dir.y);
                                rao::Measurement::SlopeTwoEdge {
                                    central_line: l.clone(),
                                    edge_length: teldiam / nsubx as f64,
                                    edge_separation: teldiam / nsubx as f64,
                                    gradient_axis: Vec2D::new(
                                        wfs_clocking[dir_idx].sin(),
                                        wfs_clocking[dir_idx].cos(),
                                    ),
                                    npoints: 1,
                                    altitude: gsalt,
                                }
                            })
                            .collect::<Vec<rao::Measurement>>(),
                        meas_coords
                            .iter()
                            .map(|p| {
                                let x0: f64 = (p.x * wfs_clocking[dir_idx].cos()
                                    + p.y * wfs_clocking[dir_idx].sin())
                                    * (1.0 + wfs_zoom[dir_idx])
                                    + wfs_delta[dir_idx].0;
                                let y0: f64 = (-p.x * wfs_clocking[dir_idx].sin()
                                    + p.y * wfs_clocking[dir_idx].cos())
                                    * (1.0 + wfs_zoom[dir_idx])
                                    + wfs_delta[dir_idx].1;
                                let l = Line::new(x0, dir.x, y0, dir.y);
                                rao::Measurement::SlopeTwoEdge {
                                    central_line: l.clone(),
                                    edge_length: teldiam / nsubx as f64,
                                    edge_separation: teldiam / nsubx as f64,
                                    gradient_axis: Vec2D::new(
                                        wfs_clocking[dir_idx].cos(),
                                        -wfs_clocking[dir_idx].sin(),
                                    ),
                                    npoints: 1,
                                    altitude: gsalt,
                                }
                            })
                            .collect::<Vec<rao::Measurement>>(),
                    ]
                })
                .flatten()
                .collect();

            let mut meas_lines = meas
                .iter()
                .map(|m| match m {
                    rao::Measurement::Zero => Line::new_on_axis(0.0, 0.0),
                    rao::Measurement::Phase { line } => line.clone(),
                    rao::Measurement::SlopeTwoLine { .. } => todo!(),
                    rao::Measurement::SlopeTwoEdge { central_line, .. } => central_line.clone(),
                })
                .collect();

            self.meas.append(&mut meas);
            self.meas_lines.append(&mut meas_lines);
        }

        fn add_com(
            &mut self,
            pitch: f64,
            nactux: u32,
            dm_delta: (f64, f64),
            dmalt: f64,
            coupling: f64,
            dm_clocking: f64,
            dm_zoom: f64,
            microns_per_volt: f64,
        ) {
            /////////////
            // define actuator related coordinates:
            let xx = Vec2D::linspace(
                &Vec2D::new(-pitch * (nactux as f64 - 1.0) * 0.5, 0.0),
                &Vec2D::new(pitch * (nactux as f64 - 1.0) * 0.5, 0.0),
                nactux,
            );
            let yy = Vec2D::linspace(
                &Vec2D::new(0.0, -pitch * (nactux as f64 - 1.0) * 0.5),
                &Vec2D::new(0.0, pitch * (nactux as f64 - 1.0) * 0.5),
                nactux,
            );
            let com_coords: Vec<Vec2D> = xx
                .iter()
                .flat_map(|x| yy.iter().map(|y| x + y).collect::<Vec<Vec2D>>())
                .collect();
            let mut com: Vec<rao::Actuator> = com_coords
                .iter()
                .map(move |p| {
                    let x: f64 = (p.x * dm_clocking.cos() + p.y * dm_clocking.sin())
                        * (1.0 + dm_zoom)
                        + dm_delta.0;
                    let y: f64 = (-p.x * dm_clocking.sin() + p.y * dm_clocking.cos())
                        * (1.0 + dm_zoom)
                        + dm_delta.1;
                    rao::Actuator::Gaussian {
                        position: Vec3D::new(x, y, dmalt),
                        sigma: rao::coupling_to_sigma(coupling, pitch),
                        microns_per_volt,
                    }
                })
                .collect();
            self.com.append(&mut com);
        }

        fn add_ttdm(&mut self, scale: f64) {
            let mut com: Vec<rao::Actuator> = vec![
                rao::Actuator::TipTilt {
                    unit_response: Vec2D::y_unit() * scale,
                },
                rao::Actuator::TipTilt {
                    unit_response: Vec2D::x_unit() * scale,
                },
            ];
            self.com.append(&mut com);
        }

        fn add_cov_layer(&mut self, r0: f64, outer_scale: f64, alt: f64, wind_x: f64, wind_y: f64) {
            self.cov_model.push(rao::VonKarmanLayer::new(
                r0,
                outer_scale,
                alt,
                Vec2D {
                    x: wind_x,
                    y: wind_y,
                },
            ));
        }

        fn set_pupil(&mut self, teldiam: f64, cobs: f64) {
            let pupil = rao::Pupil {
                rad_outer: teldiam / 2.0,
                rad_inner: cobs * teldiam / 2.0,
                spider_thickness: 0.0, // TODO: add spider api
                spiders: vec![],
            };
            self.pupil = Some(pupil);
        }

        #[staticmethod]
        fn new(
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
        ) -> Self {
            let mut tmp = Self::new_empty();
            tmp.add_com(
                pitch,
                nactux,
                dm_delta,
                dmalt,
                coupling,
                dm_clocking,
                dm_zoom,
                microns_per_volt,
            );
            tmp.add_meas(
                teldiam,
                nsubx,
                wfs_dirs,
                gsalt,
                wfs_delta,
                wfs_clocking,
                wfs_zoom,
            );
            tmp.add_phi(teldiam, nphisamples);
            tmp.add_cov_layer(0.21575883, 60.0, 0.0, 10.0, 0.0);
            tmp.add_cov_layer(0.76709884, 60.0, 1800.0, 10.0, 1.0);
            tmp.add_cov_layer(0.59536035, 60.0, 3300.0, 12.0, -2.0);
            tmp.add_cov_layer(1.24070137, 60.0, 5800.0, 15.0, 5.0);
            tmp.add_cov_layer(1.51825277, 60.0, 7400.0, 5.0, 15.0);
            tmp.add_cov_layer(0.75553414, 60.0, 13100.0, 22.0, 11.0);
            tmp.add_cov_layer(2.062782, 60.0, 15800.0, 20.0, -8.0);
            tmp.add_ts(teldiam, ntssamples, ts_dirs);
            tmp.set_pupil(teldiam, cobs);
            tmp
        }

        #[staticmethod]
        fn merge_meas(systems: Vec<SystemGeom>) -> SystemGeom {
            SystemGeom {
                meas: systems.iter().flat_map(|sys| sys.meas.clone()).collect(),
                phi: systems[0].phi.clone(),
                ts: systems[0].ts.clone(),
                com: systems[0].com.clone(),
                cov_model: systems[0].cov_model.clone(),
                pupil: systems[0].pupil.clone(),
                meas_lines: systems
                    .iter()
                    .flat_map(|sys| sys.meas_lines.clone())
                    .collect(),
                simul_dt: systems[0].simul_dt,
                meas_dt: systems[0].meas_dt,
            }
        }

        #[staticmethod]
        fn merge_com(systems: Vec<SystemGeom>) -> SystemGeom {
            SystemGeom {
                meas: systems[0].meas.clone(),
                phi: systems[0].phi.clone(),
                ts: systems[0].ts.clone(),
                com: systems.iter().flat_map(|sys| sys.com.clone()).collect(),
                cov_model: systems[0].cov_model.clone(),
                pupil: systems[0].pupil.clone(),
                meas_lines: systems[0].meas_lines.clone(),
                simul_dt: systems[0].simul_dt,
                meas_dt: systems[0].meas_dt,
            }
        }

        fn filter_com(&mut self, valid_com: Vec<bool>) {
            SystemDefinition::filter_com(self, valid_com);
        }

        fn filter_meas(&mut self, valid_meas: Vec<bool>) {
            SystemDefinition::filter_meas(self, valid_meas);
        }

        fn reorder_meas(&mut self, order: Vec<usize>) {
            SystemDefinition::reorder_meas(self, order);
        }

        fn reorder_com(&mut self, order: Vec<usize>) {
            SystemDefinition::reorder_com(self, order);
        }

        fn imat(&self) -> Vec<Vec<f64>> {
            SystemDefinition::imat(self)
        }

        fn imat_sparse(&self, indices: Vec<(usize, usize)>) -> Vec<f64> {
            SystemDefinition::imat_sparse(self, indices)
        }

        fn pmeas(&self) -> Vec<f64> {
            SystemDefinition::pmeas(self)
        }
    }

    trait SystemDefinition: Serialize + DeserializeOwned {
        fn save_yaml(&self, filename: &str) {
            fs::write(filename, yaml_serde::to_string(self).unwrap()).unwrap();
        }

        fn load_yaml(filename: &str) -> Self {
            yaml_serde::from_str(&fs::read_to_string(filename).unwrap()).unwrap()
        }

        fn com(&self) -> &Vec<Actuator>;
        fn com_mut(&mut self) -> &mut Vec<Actuator>;
        fn meas(&self) -> &Vec<Measurement>;
        fn meas_mut(&mut self) -> &mut Vec<Measurement>;
        fn meas_lines(&self) -> &Vec<Line>;
        fn meas_lines_mut(&mut self) -> &mut Vec<Line>;
        fn pupil(&self) -> Option<&Pupil>;

        fn filter_com(&mut self, valid_com: Vec<bool>) {
            *self.com_mut() = self
                .com()
                .iter()
                .enumerate()
                .filter(|(i, _)| valid_com[*i])
                .map(|(_, com)| com.clone())
                .collect();
        }

        fn filter_meas(&mut self, valid_meas: Vec<bool>) {
            *self.meas_mut() = self
                .meas()
                .iter()
                .enumerate()
                .filter(|(i, _)| valid_meas[*i])
                .map(|(_, meas)| meas.clone())
                .collect();
            *self.meas_lines_mut() = self
                .meas_lines()
                .iter()
                .enumerate()
                .filter(|(i, _)| valid_meas[*i])
                .map(|(_, meas_lines)| meas_lines.clone())
                .collect();
        }

        fn reorder_meas(&mut self, order: Vec<usize>) {
            let mut meas_new: Vec<Measurement> = vec![];
            let mut meas_lines_new: Vec<Line> = vec![];
            order.iter().for_each(|idx| {
                meas_new.push(self.meas()[*idx].clone());
                meas_lines_new.push(self.meas_lines()[*idx].clone());
            });
            *self.meas_mut() = meas_new;
            *self.meas_lines_mut() = meas_lines_new;
        }

        fn reorder_com(&mut self, order: Vec<usize>) {
            let mut com_new: Vec<Actuator> = vec![];
            order.iter().for_each(|idx| {
                com_new.push(self.com()[*idx].clone());
            });
            *self.com_mut() = com_new;
        }

        fn imat(&self) -> Vec<Vec<f64>> {
            IMat::new(self.meas(), self.com()).matrix()
        }

        fn imat_sparse(&self, indices: Vec<(usize, usize)>) -> Vec<f64> {
            IMat::new(self.meas(), self.com()).samples(indices)
        }

        fn pmeas(&self) -> Vec<f64> {
            match self.pupil() {
                Some(pupil) => IMat::new(
                    &self
                        .meas_lines()
                        .iter()
                        .flat_map(|ell| vec![Measurement::Phase { line: ell.clone() }])
                        .collect::<Vec<Measurement>>(),
                    &[pupil.clone()],
                )
                .flattened_array(),
                None => (0..self.meas_lines().len()).map(|_| 1.0).collect(),
            }
        }
    }

    #[pyclass(get_all)]
    pub struct ReconMatrices {
        pub c_ts_meas: Vec<Vec<f64>>,
        pub c_meas_meas: Vec<Vec<f64>>,
        pub d_ts_com: Vec<Vec<f64>>,
        pub d_meas_com: Vec<Vec<f64>>,
        pub p_meas: Vec<f64>,
    }

    #[pymethods]
    impl ReconMatrices {
        #[staticmethod]
        fn new(system_geom: &SystemGeom) -> Self {
            let c_meas_meas = CovMat::new(
                &system_geom.meas,
                &system_geom.meas,
                &VonKarmanLayers {
                    layers: system_geom.cov_model.clone(),
                },
                0.0,
            )
            .matrix();
            let c_ts_meas = CovMat::new(
                &system_geom.ts,
                &system_geom.meas,
                &VonKarmanLayers {
                    layers: system_geom.cov_model.clone(),
                },
                system_geom.meas_dt,
            )
            .matrix();
            let d_meas_com = IMat::new(&system_geom.meas, &system_geom.com).matrix();
            let d_ts_com = IMat::new(&system_geom.ts, &system_geom.com).matrix();
            let p_meas = match &system_geom.pupil {
                Some(pupil) => IMat::new(
                    &system_geom
                        .meas_lines
                        .iter()
                        .flat_map(|ell| {
                            vec![
                                Measurement::Phase { line: ell.clone() },
                                Measurement::Phase { line: ell.clone() },
                            ]
                        })
                        .collect::<Vec<Measurement>>(),
                    &[pupil.clone()],
                )
                .flattened_array(),
                None => (0..system_geom.meas_lines.len()).map(|_| 1.0).collect(),
            };
            ReconMatrices {
                c_meas_meas,
                c_ts_meas,
                d_ts_com,
                d_meas_com,
                p_meas,
            }
        }
    }

    #[pyclass(get_all)]
    pub struct SystemMatrices {
        pub c_phi_phi: Vec<Vec<f64>>,
        pub c_phip1_phi: Vec<Vec<f64>>,
        pub c_meas_phi: Vec<Vec<f64>>,
        pub d_meas_com: Vec<Vec<f64>>,
        pub d_phi_com: Vec<Vec<f64>>,
        pub p_phi: Vec<f64>,
        pub p_meas: Vec<f64>,
    }

    #[pymethods]
    impl SystemMatrices {
        #[staticmethod]
        fn new(system_geom: SystemGeom) -> Self {
            let c_phi_phi = CovMat::new(
                &system_geom.phi,
                &system_geom.phi,
                &VonKarmanLayers {
                    layers: system_geom.cov_model.clone(),
                },
                0.0,
            )
            .matrix();
            let c_phip1_phi = CovMat::new(
                &system_geom.phi,
                &system_geom.phi,
                &VonKarmanLayers {
                    layers: system_geom.cov_model.clone(),
                },
                system_geom.simul_dt,
            )
            .matrix();
            let c_meas_phi = CovMat::new(
                &system_geom.meas,
                &system_geom.phi,
                &VonKarmanLayers {
                    layers: system_geom.cov_model.clone(),
                },
                0.0,
            )
            .matrix();
            let d_meas_com = IMat::new(&system_geom.meas, &system_geom.com).matrix();
            let d_phi_com = IMat::new(&system_geom.phi, &system_geom.com).matrix();
            let p_phi = match &system_geom.pupil {
                Some(pupil) => IMat::new(&system_geom.phi, &[pupil.clone()]).flattened_array(),
                None => (0..system_geom.phi.len()).map(|_| 1.0).collect(),
            };
            let p_meas = match &system_geom.pupil {
                Some(pupil) => IMat::new(
                    &system_geom
                        .meas_lines
                        .into_iter()
                        .flat_map(|ell| {
                            vec![
                                Measurement::Phase { line: ell.clone() },
                                Measurement::Phase { line: ell },
                            ]
                        })
                        .collect::<Vec<Measurement>>(),
                    &[pupil.clone()],
                )
                .flattened_array(),
                None => (0..system_geom.meas.len()).map(|_| 1.0).collect(),
            };
            SystemMatrices {
                c_phi_phi,
                c_phip1_phi,
                c_meas_phi,
                d_meas_com,
                d_phi_com,
                p_phi,
                p_meas,
            }
        }
    }
}
