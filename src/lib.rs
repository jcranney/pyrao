use pyo3::prelude::*;
use rao::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn example_func_rust() -> PyResult<()> {
    println!("this is a rust function!!");
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrao(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(example_func_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ultimatestart_system_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(ultimatestart_recon_matrices, m)?)?;
    m.add_class::<SystemMatrices>()?;
    m.add_class::<ReconMatrices>()?;
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

struct SystemGeom {
    meas: Vec<Measurement>,
    phi: Vec<Measurement>,
    phip1: Vec<Measurement>,
    ts: Vec<Measurement>,
    com: Vec<Actuator>,
    cov_model: VonKarmanLayers,
    pupil: Pupil,
    meas_lines: Vec<Line>,
}

impl SystemGeom {
    fn ultimate_start() -> SystemGeom {
        const AS2RAD: f64 = 4.848e-6;
        const NPHISAMPLES: u32 = 64;
        const NTSSAMPLES: u32 = 64;
        const NSUBX: u32 = 32;
        const NACTUX: u32 = 65;
        
        /////////////
        // define phi related coordinates:
        let xx = Vec2D::linspace(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NPHISAMPLES,
        );
        let yy = Vec2D::linspace(
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
        let xx = Vec2D::linspace(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NTSSAMPLES,
        );
        let yy = Vec2D::linspace(
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
        let xx = Vec2D::linspace(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NSUBX,
        );
        let yy = Vec2D::linspace(
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
        let xx = Vec2D::linspace(
            &Vec2D::new(-4.0, 0.0),
            &Vec2D::new( 4.0, 0.0),
            NACTUX,
        );
        let yy = Vec2D::linspace(
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
            phip1,
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

impl ReconMatrices {
    fn new(system_geom: SystemGeom) -> Self {
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
            &system_geom.meas_lines.into_iter().flat_map(|ell|
            vec![
                Measurement::Phase { line: ell.clone() },
                Measurement::Phase { line: ell },
            ]).collect::<Vec<Measurement>>(),
            &[system_geom.pupil],
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
    pub c_phip1_phi: Vec<Vec<f64>>,
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
        let c_phip1_phi = CovMat::new(
            &system_geom.phip1,
            &system_geom.phi,
            &system_geom.cov_model
        ).matrix();
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
fn ultimatestart_recon_matrices() -> PyResult<ReconMatrices> {
    let system_geom = SystemGeom::ultimate_start();
    let recon_matrices = ReconMatrices::new(system_geom);
    Ok(recon_matrices)
}

#[pyfunction]
fn ultimatestart_system_matrices() -> PyResult<SystemMatrices> {
    let system_geom = SystemGeom::ultimate_start();
    let system_matrices = SystemMatrices::new(system_geom);
    Ok(system_matrices)
}
