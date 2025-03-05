use pyo3::prelude::*;
use rao::*;

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
struct SystemGeom {
    meas: Vec<Measurement>,
    phi: Vec<Measurement>,
    phip1: Option<Vec<Measurement>>,
    ts: Vec<Measurement>,
    com: Vec<Actuator>,
    cov_model: VonKarmanLayers,
    pupil: Pupil,
    meas_lines: Vec<Line>,
}

#[pymethods]
impl SystemGeom {
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
        dm_delta: (f64, f64),  // dm position offset
    ) -> Self {
        const AS2RAD: f64 = 4.848e-6;

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
        
        let phi: Vec<rao::Measurement> = phi_coords
        .iter()
        .map(|p0|
            rao::Measurement::Phase{
                line: Line::new_on_axis(p0.x,p0.y)
            }
        ).collect();
        
        /////////////
        // define truth sensor related coordinates:
        let xx = Vec2D::linspread(
            &Vec2D::new(-teldiam*0.5, 0.0),
            &Vec2D::new( teldiam*0.5, 0.0),
            ntssamples,
        );
        let yy = Vec2D::linspread(
            &Vec2D::new( 0.0, -teldiam*0.5),
            &Vec2D::new( 0.0,  teldiam*0.5),
            ntssamples,
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

        let meas: Vec<rao::Measurement> = _wfs_dirs.iter().map(|dir_arcsec|
            dir_arcsec * AS2RAD
        ).flat_map(|dir|
            vec![
                meas_coords.iter().map(move |p| {
                    let l = Line::new(p.x, dir.x, p.y, dir.y);
                    rao::Measurement::SlopeTwoEdge{
                        central_line: l.clone(),
                        edge_length: teldiam / nsubx as f64,
                        edge_separation: teldiam / nsubx as f64,
                        gradient_axis: Vec2D::y_unit(),
                        npoints: 2,
                    }
                }).collect::<Vec<rao::Measurement>>(),
                meas_coords.iter().map(move |p| {
                    let l = Line::new(p.x, dir.x, p.y, dir.y);
                    rao::Measurement::SlopeTwoEdge{
                        central_line: l.clone(),
                        edge_length: teldiam / nsubx as f64,
                        edge_separation: teldiam / nsubx as f64,
                        gradient_axis: Vec2D::x_unit(),
                        npoints: 2,
                    }
                }).collect::<Vec<rao::Measurement>>(),
            ]
        ).flatten().collect();

        let meas_lines = meas.iter().map(|m|
            match m {
                rao::Measurement::Zero => Line::new_on_axis(0.0, 0.0),
                rao::Measurement::Phase { line } => line.clone(),
                rao::Measurement::SlopeTwoLine { .. } => todo!(),
                rao::Measurement::SlopeTwoEdge { central_line, .. } => central_line.clone(),
            }
        ).collect();

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
        let com: Vec<rao::Actuator> = com_coords
        .iter()
        .map(move |p|
            rao::Actuator::Gaussian{
                position: Vec3D::new(p.x+dm_delta.0, p.y+dm_delta.1, dmalt),
                sigma: rao::coupling_to_sigma(coupling, teldiam/(nactux as f64)),
            }
        ).collect();


        let cov_model = VonKarmanLayers{
            layers: vec![
                rao::VonKarmanLayer::new(r0, 25.0, 0.0) 
            ]
        };

        let pupil = rao::Pupil {
            rad_outer: teldiam/2.0,
            rad_inner: cobs*teldiam/2.0,
            spider_thickness: 0.0,  // TODO: add spider api
            spiders: vec![],
        };
        
        SystemGeom {
            meas,
            phi,
            phip1: None,
            ts,
            com,
            cov_model,
            pupil,
            meas_lines,
        }
    }

    fn imat(&self) -> Vec<Vec<f64>> {
        IMat::new(
            &self.meas,
            &self.com
        ).matrix()
    }

    fn pmeas(&self) -> Vec<f64> {
        IMat::new(
            &self.meas_lines.iter().flat_map(|ell|
            vec![
                rao::Measurement::Phase { line: ell.clone() },
            ]).collect::<Vec<rao::Measurement>>(),
            &[self.pupil.clone()],
        ).flattened_array()
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
                },
                rao::Measurement::SlopeTwoEdge{
                    central_line: l.clone(),
                    edge_length: TELDIAM / NSUBX as f64,
                    edge_separation: TELDIAM / NSUBX as f64,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
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


        let cov_model = VonKarmanLayers{
            layers: vec![
                VonKarmanLayer::new(0.166, 25.0, 0.0)
            ]
        };

        let pupil = Pupil {
            rad_outer: TELDIAM/2.0,
            rad_inner: 0.164*TELDIAM,
            spider_thickness: 0.0,
            spiders: vec![],
        };

        SystemGeom {
            meas,
            phi,
            phip1: Some(phip1),
            ts,
            com,
            cov_model,
            pupil,
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
                },
                Measurement::SlopeTwoEdge{
                    central_line: l.clone(),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
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


        let cov_model = VonKarmanLayers{
            layers: vec![
                VonKarmanLayer::new(0.22, 25.0, 0.0)
            ]
        };

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
            phip1: Some(phip1),
            ts,
            com,
            cov_model,
            pupil,
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
            &system_geom.cov_model
        ).matrix();
        let c_ts_meas = CovMat::new(
            &system_geom.ts,
            &system_geom.meas,
            &system_geom.cov_model
        ).matrix();
        let d_meas_com = IMat::new(
            &system_geom.meas,
            &system_geom.com
        ).matrix();
        let d_ts_com = IMat::new(
            &system_geom.ts,
            &system_geom.com
        ).matrix();
        let p_meas = IMat::new(
            &system_geom.meas_lines.iter().flat_map(|ell|
            vec![
                Measurement::Phase { line: ell.clone() },
                Measurement::Phase { line: ell.clone() },
            ]).collect::<Vec<Measurement>>(),
            &[system_geom.pupil.clone()],
        ).flattened_array();
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
    pub c_phip1_phi: Option<Vec<Vec<f64>>>,
    pub c_meas_phi: Vec<Vec<f64>>,
    pub d_meas_com: Vec<Vec<f64>>,
    pub d_phi_com: Vec<Vec<f64>>,
    pub p_phi: Vec<f64>,
    pub p_meas: Vec<f64>,
}

impl SystemMatrices {
    fn new(system_geom: SystemGeom) -> Self {
        let c_phi_phi = CovMat::new(
            &system_geom.phi,
            &system_geom.phi,
            &system_geom.cov_model
        ).matrix();
        let c_phip1_phi = system_geom.phip1.map(|phip1|
            CovMat::new(
                &phip1,
                &system_geom.phi,
                &system_geom.cov_model
            ).matrix()
        );
        let c_meas_phi = CovMat::new(
            &system_geom.meas,
            &system_geom.phi,
            &system_geom.cov_model
        ).matrix();
        let d_meas_com = IMat::new(
            &system_geom.meas,
            &system_geom.com
        ).matrix();
        let d_phi_com = IMat::new(
            &system_geom.phi,
            &system_geom.com
        ).matrix();
        let pup = vec![system_geom.pupil];
        let p_phi = IMat::new(
            &system_geom.phi,
            &pup,
        ).flattened_array();
        let p_meas = IMat::new(
            &system_geom.meas_lines.into_iter().flat_map(|ell|
            vec![
                Measurement::Phase { line: ell.clone() },
                Measurement::Phase { line: ell },
            ]).collect::<Vec<Measurement>>(),
            &pup,
        ).flattened_array();
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
