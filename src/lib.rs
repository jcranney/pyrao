use pyo3::prelude::*;
use rao::*;

const AS2RAD: f64 = 4.848e-6;

/// A Python module implemented in Rust.
#[pymodule]
fn pyrao(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ultimatestart_system_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(ultimatestart_recon_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(gems_recon_matrices, m)?)?;
    m.add_class::<SystemMatrices>()?;
    m.add_class::<ReconMatrices>()?;
    m.add_class::<SystemGeom>()?;
    Ok(())
}

#[derive(Debug,Clone)]
struct VonKarmanLayers {
    layers: Vec<VonKarmanLayer>
}

impl CoSampleable for VonKarmanLayers {
    fn cosample(&self, p: &Line, q: &Line) -> f64 {
        self.layers.iter().map(
            |layer| layer.cosample(p, q)
        ).sum()
    }
}

#[pyclass]
#[derive(Clone)]
struct SystemGeom {
    meas: Vec<Measurement>,
    phi: Vec<Measurement>,
    phip1: Vec<Measurement>,
    ts: Vec<Measurement>,
    com: Vec<Actuator>,
    cov_model: Vec<VonKarmanLayer>,
    pupil: Option<Pupil>,
    meas_lines: Vec<Line>,
}

#[pymethods]
impl SystemGeom {
    #[staticmethod]
    fn new_empty() -> Self {
        Self { 
            meas: vec![],
            phi: vec![],
            phip1: vec![],
            ts: vec![],
            com: vec![],
            cov_model: vec![],
            pupil: None,
            meas_lines: vec![],
        }
    }

    fn add_phi(&mut self, teldiam: f64, nphisamples: u32) {
        /////////////
        // define phi related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-teldiam*0.5, 0.0),
            &Vec2D::new( teldiam*0.5, 0.0),
            nphisamples,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -teldiam*0.5),
            &Vec2D::new( 0.0,  teldiam*0.5),
            nphisamples,
        );
        let phi_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x|
            yy.iter().map(move |y| {
                x+y
            })).collect();
        
        let mut phi: Vec<rao::Measurement> = phi_coords
        .iter()
        .map(|p0|
            rao::Measurement::Phase{
                line: Line::new_on_axis(p0.x,p0.y)
            }
        ).collect();
        
        self.phi.append(&mut phi);
    }

    fn add_ts(&mut self, teldiam: f64, ntssamples: u32, ts_dirs: Vec<(f64, f64)>) {
        /////////////
        // define truth sensor related coordinates:
        let xx = Vec2D::linspace(
            &Vec2D::new(-teldiam*0.5, 0.0),
            &Vec2D::new( teldiam*0.5, 0.0),
            ntssamples,
        );
        let yy = Vec2D::linspace(
            &Vec2D::new( 0.0, -teldiam*0.5),
            &Vec2D::new( 0.0,  teldiam*0.5),
            ntssamples,
        );
        let ts_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x|
            yy.iter().map(move |y| {
                x+y
            })).collect();
        
        let mut ts: Vec<Measurement> = ts_dirs.into_iter().map(|(x_as,y_as)|
            ts_coords
            .iter()
            .map(move |p0|
                rao::Measurement::Phase{
                    line: Line::new(p0.x, x_as*AS2RAD, p0.y, y_as*AS2RAD)
                }
            )
        ).flatten().collect();
        self.ts.append(&mut ts);
    }
    
    fn add_meas(
        &mut self, teldiam: f64, nsubx: u32, wfs_dirs: Vec<(f64, f64)>, 
        gsalt: f64, wfs_delta: Vec<(f64, f64)>, wfs_clocking: Vec<f64>, 
        wfs_zoom: Vec<f64>
    ) {
        /////////////
        // define rao::Measurement related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-teldiam*0.5, 0.0),
            &Vec2D::new( teldiam*0.5, 0.0),
            nsubx,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -teldiam*0.5),
            &Vec2D::new( 0.0,  teldiam*0.5),
            nsubx,
        );
        let meas_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x| 
            yy.iter().map(|y| {
                x+y
            }).collect::<Vec<Vec2D>>()).collect();
        let _wfs_dirs: Vec<Vec2D> = wfs_dirs.into_iter().map(|(x,y)|
            Vec2D::new(x, y)
        ).collect();

        let mut meas: Vec<rao::Measurement> = _wfs_dirs.iter().enumerate().map(|(dir_idx, dir_arcsec)|
            (dir_idx, dir_arcsec * AS2RAD)
        ).flat_map(|(dir_idx, dir)|
            vec![
                meas_coords.iter().map(|p| {
                    let x0: f64 = (
                        p.x * wfs_clocking[dir_idx].cos() + p.y * wfs_clocking[dir_idx].sin()
                    ) * (1.0 + wfs_zoom[dir_idx]) + wfs_delta[dir_idx].0;
                    let y0: f64 = (
                        - p.x * wfs_clocking[dir_idx].sin() + p.y * wfs_clocking[dir_idx].cos()
                    ) * (1.0 + wfs_zoom[dir_idx]) + wfs_delta[dir_idx].1;
                    let l = Line::new(x0, dir.x, y0, dir.y);
                    rao::Measurement::SlopeTwoEdge{
                        central_line: l.clone(),
                        edge_length: teldiam / nsubx as f64,
                        edge_separation: teldiam / nsubx as f64,
                        gradient_axis: Vec2D::new(wfs_clocking[dir_idx].sin(), wfs_clocking[dir_idx].cos()),
                        npoints: 1,
                        altitude: gsalt,
                    }
                }).collect::<Vec<rao::Measurement>>(),
                meas_coords.iter().map(|p| {
                    let x0: f64 = (
                        p.x * wfs_clocking[dir_idx].cos() + p.y * wfs_clocking[dir_idx].sin()
                    ) * (1.0 + wfs_zoom[dir_idx]) + wfs_delta[dir_idx].0;
                    let y0: f64 = (
                        - p.x * wfs_clocking[dir_idx].sin() + p.y * wfs_clocking[dir_idx].cos()
                    ) * (1.0 + wfs_zoom[dir_idx]) + wfs_delta[dir_idx].1;
                    let l = Line::new(x0, dir.x, y0, dir.y);
                    rao::Measurement::SlopeTwoEdge{
                        central_line: l.clone(),
                        edge_length: teldiam / nsubx as f64,
                        edge_separation: teldiam / nsubx as f64,
                        gradient_axis: Vec2D::new(wfs_clocking[dir_idx].cos(), - wfs_clocking[dir_idx].sin()),
                        npoints: 1,
                        altitude: gsalt,
                    }
                }).collect::<Vec<rao::Measurement>>(),
            ]
        ).flatten().collect();

        let mut meas_lines = meas.iter().map(|m|
            match m {
                rao::Measurement::Zero => Line::new_on_axis(0.0, 0.0),
                rao::Measurement::Phase { line } => line.clone(),
                rao::Measurement::SlopeTwoLine { .. } => todo!(),
                rao::Measurement::SlopeTwoEdge { central_line, .. } => central_line.clone(),
            }
        ).collect();

        self.meas.append(&mut meas);
        self.meas_lines.append(&mut meas_lines);
    }

    fn add_com(
        &mut self, pitch: f64, nactux: u32, dm_delta: (f64, f64), dmalt: f64, 
        coupling: f64, dm_clocking: f64, dm_zoom: f64
    ) {
        /////////////
        // define actuator related coordinates:
        let xx = Vec2D::linspace(
            &Vec2D::new(-pitch * (nactux as f64 - 1.0) * 0.5, 0.0),
            &Vec2D::new( pitch * (nactux as f64 - 1.0) * 0.5, 0.0),
            nactux,
        );
        let yy = Vec2D::linspace(
            &Vec2D::new(0.0, -pitch * (nactux as f64 - 1.0) * 0.5),
            &Vec2D::new(0.0,  pitch * (nactux as f64 - 1.0) * 0.5),
            nactux,
        );
        let com_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x| 
            yy.iter().map(|y| {
                x+y
            }).collect::<Vec<Vec2D>>()).collect();
        let mut com: Vec<rao::Actuator> = com_coords
        .iter()
        .map(move |p| {
            let x: f64 = (
                p.x * dm_clocking.cos() + p.y * dm_clocking.sin()
            ) * (1.0 + dm_zoom) + dm_delta.0;
            let y: f64 = (
                - p.x * dm_clocking.sin() + p.y * dm_clocking.cos()
            ) * (1.0 + dm_zoom) + dm_delta.1;
            rao::Actuator::Gaussian{
                position: Vec3D::new(x, y, dmalt),
                sigma: rao::coupling_to_sigma(coupling, pitch),
            }
        }).collect();
        self.com.append(&mut com);
    }
    
    fn add_ttdm(&mut self, scale: f64) {
        let mut com: Vec<rao::Actuator> = vec![
            rao::Actuator::TipTilt { unit_response: Vec2D::y_unit() * scale},
            rao::Actuator::TipTilt { unit_response: Vec2D::x_unit() * scale},
        ];
        self.com.append(&mut com);
    }

    fn add_cov_layer(&mut self) {
        let mut cov_model = 
            vec![
                rao::VonKarmanLayer::new(0.21575883, 60.0, 0.0),
                rao::VonKarmanLayer::new(0.76709884, 60.0, 1800.0),
                rao::VonKarmanLayer::new(0.59536035, 60.0, 3300.0),
                rao::VonKarmanLayer::new(1.24070137, 60.0, 5800.0),
                rao::VonKarmanLayer::new(1.51825277, 60.0, 7400.0),
                rao::VonKarmanLayer::new(0.75553414, 60.0, 13100.0),
                rao::VonKarmanLayer::new(2.062782, 60.0, 15800.0),
            ];
        self.cov_model.append(&mut cov_model);
    }

    fn set_pupil(&mut self, teldiam: f64, cobs: f64) {
        let pupil = rao::Pupil {
            rad_outer: teldiam/2.0,
            rad_inner: cobs*teldiam/2.0,
            spider_thickness: 0.0,  // TODO: add spider api
            spiders: vec![],
        };
        self.pupil = Some(pupil);
    }

    #[staticmethod]
    fn new(
        teldiam: f64,  // diameter of telescope in metres
        cobs: f64,  // central obscuration, fraction of diameter
        r0: f64,  // seeing (metres)
        coupling: f64,  // coupling between DM actuators
        nactux: u32,  // number of actuators across DM diameter
        dmalt: f64,  // dm altitude in metres
        pitch: f64,  // dm pitch in metres
        nsubx: u32,  // number of subapertures across WFS pupil
        ntssamples: u32,  // number of samples across TS pupil
        nphisamples: u32,  // number of phase samples across pupil,
        wfs_dirs: Vec<(f64, f64)>,  // directions of WFSs (arcsec)
        ts_dirs: Vec<(f64, f64)>,  // directions of WFSs (arcsec)
        dm_delta: (f64, f64),  // dm position offset (metres)
        wfs_delta: Vec<(f64, f64)>,  // wfs position offset (metres)
        dm_clocking: f64,  // dm rotation (radians)
        wfs_clocking: Vec<f64>,  // wfs rotation (radians)
        dm_zoom: f64,  // dm magnification error (0.0 === unity magnification)
        wfs_zoom: Vec<f64>,  // wfs magnification error (0.0 === unity magnification)
        gsalt: f64,  // guide star altitude
    ) -> Self {
        let mut tmp = Self::new_empty();
        tmp.add_com(pitch, nactux, dm_delta, dmalt, coupling, dm_clocking, dm_zoom);
        tmp.add_meas(teldiam, nsubx, wfs_dirs, gsalt, wfs_delta, wfs_clocking, wfs_zoom);
        tmp.add_phi(teldiam, nphisamples);
        tmp.add_cov_layer();
        tmp.add_ts(teldiam, ntssamples, ts_dirs);
        tmp.set_pupil(teldiam, cobs);
        tmp
    }

    #[staticmethod]
    fn merge_meas(systems: Vec<SystemGeom>) -> SystemGeom {
        SystemGeom { 
            meas: systems.iter().flat_map(|sys| sys.meas.clone()).collect(), 
            phi: systems[0].phi.clone(),
            phip1: systems[0].phip1.clone(),
            ts: systems[0].ts.clone(),
            com: systems[0].com.clone(),
            cov_model: systems[0].cov_model.clone(),
            pupil: systems[0].pupil.clone(),
            meas_lines: systems.iter().flat_map(|sys| sys.meas_lines.clone()).collect(),
        }
    }

    #[staticmethod]
    fn merge_com(systems: Vec<SystemGeom>) -> SystemGeom {
        SystemGeom { 
            meas: systems[0].meas.clone(),
            phi: systems[0].phi.clone(),
            phip1: systems[0].phip1.clone(),
            ts: systems[0].ts.clone(),
            com: systems.iter().flat_map(|sys| sys.com.clone()).collect(),
            cov_model: systems[0].cov_model.clone(),
            pupil: systems[0].pupil.clone(),
            meas_lines: systems[0].meas_lines.clone(),
        }
    }

    fn filter_com(&mut self, valid_com: Vec<bool>) {
        self.com = self.com.iter()
        .enumerate()
        .filter(|(i, _)| valid_com[*i])
        .map(|(_,com)| com.clone())
        .collect();
    }

    fn filter_meas(&mut self, valid_meas: Vec<bool>) {
        self.meas = self.meas.iter()
        .enumerate()
        .filter(|(i, _)| valid_meas[*i])
        .map(|(_,meas)| meas.clone())
        .collect();
        self.meas_lines = self.meas_lines.iter()
        .enumerate()
        .filter(|(i, _)| valid_meas[*i])
        .map(|(_,meas_lines)| meas_lines.clone())
        .collect();

    }

    fn imat(&self) -> Vec<Vec<f64>> {
        IMat::new(
            &self.meas,
            &self.com
        ).matrix()
    }
    
    fn imat_sparse(&self, indices: Vec<(usize, usize)>) -> Vec<f64> {
        IMat::new(
            &self.meas,
            &self.com
        ).samples(indices)
    }

    fn pmeas(&self) -> Vec<f64> {
        match &self.pupil {
            Some(pupil) => {
                IMat::new(
                    &self.meas_lines.iter().flat_map(|ell|
                    vec![
                        Measurement::Phase { line: ell.clone() },
                    ]).collect::<Vec<Measurement>>(),
                    &[pupil.clone()]
                ).flattened_array()
            },
            None => (0..self.meas_lines.len()).map(|_| 1.0).collect(),
        }
    }
}

impl SystemGeom {
    fn gems() -> SystemGeom {
        const AS2RAD: f64 = 4.848e-6;
        const NPHISAMPLES: u32 = 64;
        const NTSSAMPLES: u32 = 64;
        const NSUBX: u32 = 16;
        const NACTUX: u32 = 17;
        const TELDIAM: f64 = 7.9; // metres
        /////////////
        // define phi related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-TELDIAM*0.5, 0.0),
            &Vec2D::new( TELDIAM*0.5, 0.0),
            NPHISAMPLES,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -TELDIAM*0.5),
            &Vec2D::new( 0.0,  TELDIAM*0.5),
            NPHISAMPLES,
        );
        let phi_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x|
            yy.iter().map(move |y| {
                x+y
            })).collect();
        
        let phi: Vec<rao::Measurement> = phi_coords
        .iter()
        .map(|p0|
            rao::Measurement::Phase{
                line: Line::new_on_axis(p0.x,p0.y)
            }
        ).collect();
        
        let phip1: Vec<rao::Measurement> = phi_coords
        .iter()
        .map(|p0|
            rao::Measurement::Phase{
                line: Line::new_on_axis(p0.x+0.005,p0.y)
            }
        ).collect();
        
        /////////////
        // define truth sensor related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-TELDIAM*0.5, 0.0),
            &Vec2D::new( TELDIAM*0.5, 0.0),
            NTSSAMPLES,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -TELDIAM*0.5),
            &Vec2D::new( 0.0,  TELDIAM*0.5),
            NTSSAMPLES,
        );
        let ts_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x|
            yy.iter().map(move |y| {
                x+y
            })).collect();
        
        let ts: Vec<rao::Measurement> = ts_coords
        .iter()
        .map(|p0|
            rao::Measurement::Phase{
                line: Line::new_on_axis(p0.x,p0.y)
            }
        ).collect();
        
        /////////////
        // define measurement related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-TELDIAM*0.5, 0.0),
            &Vec2D::new( TELDIAM*0.5, 0.0),
            NSUBX,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -TELDIAM*0.5),
            &Vec2D::new( 0.0,  TELDIAM*0.5),
            NSUBX,
        );
        let meas_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x| 
            yy.iter().map(|y| {
                x+y
            }).collect::<Vec<Vec2D>>()).collect();
        let wfs_dirs = vec![
            Vec2D::new(  0.0,   0.0),
            Vec2D::new(-30.0, -30.0),
            Vec2D::new( 30.0, -30.0),
            Vec2D::new( 30.0,  30.0),
            Vec2D::new(-30.0,  30.0),
        ];
        let meas_lines: Vec<Line> = wfs_dirs.into_iter()
        .map(|dir_arcsec|
            dir_arcsec * AS2RAD
        ).flat_map(|dir|
            meas_coords
            .iter().map(move |p|
                Line::new(p.x, dir.x, p.y, dir.y)
            )
        ).collect();

        let meas: Vec<rao::Measurement> = meas_lines.iter()
        .flat_map(|l|
            vec![
                rao::Measurement::SlopeTwoEdge{
                    central_line: l.clone(),
                    edge_length: TELDIAM / NSUBX as f64,
                    edge_separation: TELDIAM / NSUBX as f64,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                    altitude: 90000.0,
                },
                rao::Measurement::SlopeTwoEdge{
                    central_line: l.clone(),
                    edge_length: TELDIAM / NSUBX as f64,
                    edge_separation: TELDIAM / NSUBX as f64,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                    altitude: 90000.0,
                }
            ]).collect();

        /////////////
        // define actuator related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-TELDIAM*0.5, 0.0),
            &Vec2D::new( TELDIAM*0.5, 0.0),
            NACTUX,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -TELDIAM*0.5),
            &Vec2D::new( 0.0,  TELDIAM*0.5),
            NACTUX,
        );
        let com_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x| 
            yy.iter().map(|y| {
                x+y
            }).collect::<Vec<Vec2D>>()).collect();
        let com: Vec<rao::Actuator> = com_coords
        .iter()
        .map(move |p|
            rao::Actuator::Gaussian{
                position: Vec3D::new(p.x, p.y, 0.0),
                sigma: coupling_to_sigma(0.105, TELDIAM/(NACTUX as f64)),
            }
        ).collect();


        let cov_model = vec![
            VonKarmanLayer::new(0.166, 25.0, 0.0)
        ];

        let pupil = Pupil {
            rad_outer: TELDIAM/2.0,
            rad_inner: 0.164*TELDIAM,
            spider_thickness: 0.0,
            spiders: vec![],
        };

        SystemGeom {
            meas,
            phi,
            phip1,
            ts,
            com,
            cov_model,
            pupil: Some(pupil),
            meas_lines,
        }
    }
    fn ultimate_start() -> SystemGeom {
        const AS2RAD: f64 = 4.848e-6;
        const NPHISAMPLES: u32 = 64;
        const NTSSAMPLES: u32 = 64;
        const NSUBX: u32 = 32;
        const NACTUX: u32 = 65;
        
        /////////////
        // define phi related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NPHISAMPLES,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -4.0),
            &Vec2D::new( 0.0,  4.0),
            NPHISAMPLES,
        );
        let phi_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x|
            yy.iter().map(move |y| {
                x+y
            })).collect();
        
        let phi: Vec<Measurement> = phi_coords
        .iter()
        .map(|p0|
            Measurement::Phase{
                line: Line::new_on_axis(p0.x,p0.y)
            }
        ).collect();
        
        let phip1: Vec<Measurement> = phi_coords
        .iter()
        .map(|p0|
            Measurement::Phase{
                line: Line::new_on_axis(p0.x+0.005,p0.y)
            }
        ).collect();
        
        /////////////
        // define truth sensor related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NTSSAMPLES,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -4.0),
            &Vec2D::new( 0.0,  4.0),
            NTSSAMPLES,
        );
        let ts_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x|
            yy.iter().map(move |y| {
                x+y
            })).collect();
        
        let ts: Vec<Measurement> = ts_coords
        .iter()
        .map(|p0|
            Measurement::Phase{
                line: Line::new_on_axis(p0.x,p0.y)
            }
        ).collect();
        
        /////////////
        // define measurement related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NSUBX,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -4.0),
            &Vec2D::new( 0.0,  4.0),
            NSUBX,
        );
        let meas_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x| 
            yy.iter().map(|y| {
                x+y
            }).collect::<Vec<Vec2D>>()).collect();
        let wfs_dirs = vec![
            Vec2D::new(-10.0, -10.0),
            Vec2D::new(-10.0,  10.0),
            Vec2D::new( 10.0, -10.0),
            Vec2D::new( 10.0,  10.0),
        ];
        let meas_lines: Vec<Line> = wfs_dirs.into_iter()
        .map(|dir_arcsec|
            dir_arcsec * AS2RAD
        ).flat_map(|dir|
            meas_coords
            .iter().map(move |p|
                Line::new(p.x, dir.x, p.y, dir.y)
            )
        ).collect();

        let meas: Vec<Measurement> = meas_lines.iter()
        .flat_map(|l|
            vec![
                Measurement::SlopeTwoEdge{
                    central_line: l.clone(),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                    altitude: 90000.0,
                },
                Measurement::SlopeTwoEdge{
                    central_line: l.clone(),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                    altitude: 90000.0,
                }
            ]).collect();

        /////////////
        // define actuator related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NACTUX,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -4.0),
            &Vec2D::new( 0.0,  4.0),
            NACTUX,
        );
        let com_coords: Vec<Vec2D> = xx.iter()
        .flat_map(|x| 
            yy.iter().map(|y| {
                x+y
            }).collect::<Vec<Vec2D>>()).collect();
        let com: Vec<Actuator> = com_coords
        .iter()
        .map(move |p|
            Actuator::Gaussian{
                position: Vec3D::new(p.x, p.y, 0.0),
                sigma: coupling_to_sigma(0.3, 8.0/(NACTUX as f64 - 1.0)),
            }
        ).collect();


        let cov_model = vec![
            VonKarmanLayer::new(0.22, 25.0, 0.0)
        ];

        let pupil = Pupil {
            rad_outer: 4.1,
            rad_inner: 1.2,
            spider_thickness: 0.2,
            spiders: vec![
                (Vec2D::new(0.0,1.2), Vec2D::new(4.0,-4.0)),
                (Vec2D::new(0.0,1.2), Vec2D::new(-4.0,-4.0)),
                (Vec2D::new(0.0,-1.2), Vec2D::new(4.0,4.0)),
                (Vec2D::new(0.0,-1.2), Vec2D::new(-4.0,4.0)),
            ]
        };

        SystemGeom {
            meas,
            phi,
            phip1,
            ts,
            com,
            cov_model,
            pupil: Some(pupil),
            meas_lines,
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
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let c_ts_meas = CovMat::new(
            &system_geom.ts,
            &system_geom.meas,
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let d_meas_com = IMat::new(
            &system_geom.meas,
            &system_geom.com
        ).matrix();
        let d_ts_com = IMat::new(
            &system_geom.ts,
            &system_geom.com
        ).matrix();
        let p_meas = match &system_geom.pupil {
            Some(pupil) => {
                IMat::new(
                            &system_geom.meas_lines.iter().flat_map(|ell|
                            vec![
                                Measurement::Phase { line: ell.clone() },
                                Measurement::Phase { line: ell.clone() },
                            ]).collect::<Vec<Measurement>>(),
                            &[pupil.clone()],
                        ).flattened_array()
            },
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
    pub c_meas_meas: Vec<Vec<f64>>,
    pub c_ts_meas: Vec<Vec<f64>>,
    pub d_meas_com: Vec<Vec<f64>>,
    pub d_phi_com: Vec<Vec<f64>>,
    pub d_ts_com: Vec<Vec<f64>>,
    pub p_phi: Vec<f64>,
    pub p_meas: Vec<f64>,
    pub p_ts: Vec<f64>,
}

#[pymethods]
impl SystemMatrices {
    #[staticmethod]
    fn new(system_geom: SystemGeom) -> Self {
        let c_phi_phi = CovMat::new(
            &system_geom.phi,
            &system_geom.phi,
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let c_phip1_phi = CovMat::new(
            &system_geom.phip1,
            &system_geom.phi,
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let c_meas_phi = CovMat::new(
            &system_geom.meas,
            &system_geom.phi,
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let c_meas_meas = CovMat::new(
            &system_geom.meas,
            &system_geom.meas,
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let c_ts_meas = CovMat::new(
            &system_geom.ts,
            &system_geom.meas,
            &VonKarmanLayers{layers: system_geom.cov_model.clone()},
        ).matrix();
        let d_meas_com = IMat::new(
            &system_geom.meas,
            &system_geom.com
        ).matrix();
        let d_ts_com = IMat::new(
            &system_geom.ts,
            &system_geom.com
        ).matrix();
        let d_phi_com = IMat::new(
            &system_geom.phi,
            &system_geom.com
        ).matrix();
        let p_phi = match &system_geom.pupil {
            Some(pupil) => {
                IMat::new(
                    &system_geom.phi,
                    &[pupil.clone()],
                ).flattened_array()
            },
            None => (0..system_geom.phi.len()).map(|_| 1.0).collect(),
        };
        let p_meas = match &system_geom.pupil {
            Some(pupil) => {
                IMat::new(
                    &system_geom.meas_lines.into_iter().flat_map(|ell|
                    vec![
                        Measurement::Phase { line: ell.clone() },
                        Measurement::Phase { line: ell },
                    ]).collect::<Vec<Measurement>>(),
                    &[pupil.clone()],
                ).flattened_array()
            },
            None => (0..system_geom.meas.len()).map(|_| 1.0).collect(),
        };
        let p_ts = match &system_geom.pupil {
            Some(pupil) => {
                IMat::new(
                    &system_geom.ts,
                    &[pupil.clone()],
                ).flattened_array()
            },
            None => (0..system_geom.ts.len()).map(|_| 1.0).collect(),
        };
        SystemMatrices {
            c_phi_phi,
            c_phip1_phi,
            c_meas_phi,
            c_meas_meas,
            c_ts_meas,
            d_ts_com,
            d_meas_com,
            d_phi_com,
            p_phi,
            p_meas,
            p_ts,
        }
    }
}

#[pyfunction]
fn gems_recon_matrices() -> PyResult<ReconMatrices> {
    let system_geom = SystemGeom::gems();
    let recon_matrices = ReconMatrices::new(&system_geom);
    Ok(recon_matrices)
}

#[pyfunction]
fn ultimatestart_recon_matrices() -> PyResult<ReconMatrices> {
    let system_geom = SystemGeom::ultimate_start();
    let recon_matrices = ReconMatrices::new(&system_geom);
    Ok(recon_matrices)
}

#[pyfunction]
fn ultimatestart_system_matrices() -> PyResult<SystemMatrices> {
    let system_geom = SystemGeom::ultimate_start();
    let system_matrices = SystemMatrices::new(system_geom);
    Ok(system_matrices)
}
