use std::fs;

use pyo3::prelude::*;
use rao::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A Python module implemented in Rust.
#[pymodule]
pub mod pyrao {
    use super::*;

    #[pyclass(from_py_object)]
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

    /// The scope of a `CompactSystem` is minimal, just enough to uniquely
    /// expand into a full `ExpandedSystem` without ambiguity, including any
    /// misregistrations.
    #[pyclass(from_py_object)]
    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    struct CompactSystem {
        telescope: Telescope,
        dm: Vec<Dm>,
        wfs: Vec<Wfs>,
        ctrl: Ctrl,
        atmos: Atmos,
    }
    #[pymethods]
    impl CompactSystem {
        #[new]
        fn new(telescope: Telescope, dm: Vec<Dm>, wfs: Vec<Wfs>, ctrl: Ctrl, atmos: Atmos) -> Self {
            Self {
                telescope,
                dm,
                wfs,
                ctrl,
                atmos,
            }
        }
        fn expand(&self) -> ExpandedSystem {
            self.into()
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
        #[new]
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
        #[new]
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
        #[new]
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
        #[new]
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
            npoints: usize,
            width: f64,
            height: f64,
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
            npoints: usize,
            width: f64,
            height: f64,
        ) -> Self {
            Self::RectGrid { npoints, width, height }
        }
    }

    impl Positions {
        pub fn pitch(&self) -> (f64, f64) {
            match self {
                Positions::RectGrid { npoints, width, height } => (
                    width / (*npoints-1) as f64,
                    height / (*npoints-1) as f64,
                ),
                Positions::Explicit { pitch, .. } => *pitch,
            }
        }
    }

    impl From<&Positions> for Vec<Vec2D> {
        fn from(value: &Positions) -> Self {
            match value {
                Positions::RectGrid { npoints, width, height } => {
                    let x = Vec2D::linspace(
                        &Vec2D::new(-width/2.0, 0.0),
                        &Vec2D::new(width/2.0, 0.0),
                        *npoints as u32,
                    );
                    let y = Vec2D::linspace(
                        &Vec2D::new(0.0, -height/2.0),
                        &Vec2D::new(0.0, height/2.0),
                        *npoints as u32,
                    );
                    x.into_iter().zip(y).map(|(x, y)| x + y).collect()
                }
                Positions::Explicit { pos, .. } => {
                    pos.iter().map(|(x, y)| Vec2D::new(*x, *y)).collect()
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
        #[new]
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
        #[new]
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
        #[new]
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
        #[new]
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

    impl From<&CompactSystem> for ExpandedSystem {
        fn from(value: &CompactSystem) -> Self {
            let CompactSystem {
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
    struct ExpandedSystem {
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

    impl System for ExpandedSystem {
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
    impl ExpandedSystem {
        fn save_yaml(&self, filename: &str) {
            System::save_yaml(self, filename);
        }

        #[staticmethod]
        fn load_yaml(filename: &str) -> Self {
            System::load_yaml(filename)
        }

        // tmp.add_cov_layer(0.21575883, 60.0, 0.0, 10.0, 0.0);
        // tmp.add_cov_layer(0.76709884, 60.0, 1800.0, 10.0, 1.0);
        // tmp.add_cov_layer(0.59536035, 60.0, 3300.0, 12.0, -2.0);
        // tmp.add_cov_layer(1.24070137, 60.0, 5800.0, 15.0, 5.0);
        // tmp.add_cov_layer(1.51825277, 60.0, 7400.0, 5.0, 15.0);
        // tmp.add_cov_layer(0.75553414, 60.0, 13100.0, 22.0, 11.0);
        // tmp.add_cov_layer(2.062782, 60.0, 15800.0, 20.0, -8.0);
        fn layers(&self) -> VonKarmanLayers {
            VonKarmanLayers {
                layers: self.cov_model.clone(),
            }
        }


        fn filter_com(&mut self, valid_com: Vec<bool>) {
            System::filter_com(self, valid_com);
        }

        fn filter_meas(&mut self, valid_meas: Vec<bool>) {
            System::filter_meas(self, valid_meas);
        }

        fn reorder_meas(&mut self, order: Vec<usize>) {
            System::reorder_meas(self, order);
        }

        fn reorder_com(&mut self, order: Vec<usize>) {
            System::reorder_com(self, order);
        }

        fn p_meas(&self) -> Vec<f64> {
            System::pmeas(self)
        }

        fn c_meas_meas(&self) -> (Vec<f64>, (usize, usize)) {
            let layers = self.layers();
            let c_meas_meas = CovMat::new(&self.meas, &self.meas, &layers, 0.0);
            (
                c_meas_meas.flattened_array(),
                (c_meas_meas.nrows(), c_meas_meas.ncols()),
            )
        }

        fn c_ts_meas(&self) -> (Vec<f64>, (usize, usize)) {
            let layers = self.layers();
            let c_ts_meas = CovMat::new(&self.ts, &self.meas, &layers, self.meas_dt);
            (
                c_ts_meas.flattened_array(),
                (c_ts_meas.nrows(), c_ts_meas.ncols()),
            )
        }

        fn d_meas_com(&self) -> (Vec<f64>, (usize, usize)) {
            let d_meas_com = IMat::new(&self.meas, &self.com);
            (
                d_meas_com.flattened_array(),
                (d_meas_com.nrows(), d_meas_com.ncols()),
            )
        }

        fn d_ts_com(&self) -> (Vec<f64>, (usize, usize)) {
            let d_ts_com = IMat::new(&self.ts, &self.com);
            (
                d_ts_com.flattened_array(),
                (d_ts_com.nrows(), d_ts_com.ncols()),
            )
        }

        fn c_phi_phi(&self) -> (Vec<f64>, (usize, usize)) {
            let layers = self.layers();

            let x = CovMat::new(&self.phi, &self.phi, &layers, 0.0);
            (x.flattened_array(), (x.nrows(), x.ncols()))
        }
        fn c_phip1_phi(&self) -> (Vec<f64>, (usize, usize)) {
            let layers = self.layers();

            let x = CovMat::new(&self.phi, &self.phi, &layers, self.simul_dt);
            (x.flattened_array(), (x.nrows(), x.ncols()))
        }
        fn c_meas_phi(&self) -> (Vec<f64>, (usize, usize)) {
            let layers = self.layers();

            let x = CovMat::new(&self.meas, &self.phi, &layers, 0.0);
            (x.flattened_array(), (x.nrows(), x.ncols()))
        }
        fn d_phi_com(&self) -> (Vec<f64>, (usize, usize)) {
            let x = IMat::new(&self.phi, &self.com);
            (x.flattened_array(), (x.nrows(), x.ncols()))
        }
        fn p_phi(&self) -> Vec<f64> {
            match &self.pupil {
                Some(pupil) => IMat::new(&self.phi, std::slice::from_ref(pupil)).flattened_array(),
                None => (0..self.phi.len()).map(|_| 1.0).collect(),
            }
        }
    }

    trait System: Serialize + DeserializeOwned {
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

        fn pmeas(&self) -> Vec<f64> {
            match self.pupil() {
                Some(pupil) => IMat::new(
                    &self
                        .meas_lines()
                        .iter()
                        .flat_map(|ell| vec![Measurement::Phase { line: ell.clone() }])
                        .collect::<Vec<Measurement>>(),
                    std::slice::from_ref(pupil),
                )
                .flattened_array(),
                None => (0..self.meas_lines().len()).map(|_| 1.0).collect(),
            }
        }
    }
}

