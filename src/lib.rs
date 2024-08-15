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
    m.add_function(wrap_pyfunction!(build_system_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(ultimatestart_system_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(build_cpp, m)?)?;
    m.add_function(wrap_pyfunction!(build_cpm, m)?)?;
    m.add_function(wrap_pyfunction!(build_cmm, m)?)?;
    m.add_function(wrap_pyfunction!(build_dpc, m)?)?;
    m.add_class::<InitMatrices>()?;
    Ok(())
}

#[pyclass(get_all)]
pub struct InitMatrices {
    pub c_phi_phi: Vec<Vec<f64>>,
    pub c_phip1_phi: Vec<Vec<f64>>,
    pub c_meas_phi: Vec<Vec<f64>>,
    pub d_meas_com: Vec<Vec<f64>>,
    pub d_phi_com: Vec<Vec<f64>>,
}

#[pyfunction]
fn build_cpp() -> PyResult<Vec<Vec<f64>>> {
    const NPHISAMPLES: u32 = 32;
    
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
    .map(|x| 
        yy.iter().map(move |y| {
            x+y
        })
    ).flatten().collect();
    
    let phi: Vec<Measurement> = phi_coords
    .iter()
    .map(|p0|
        Measurement::Phase{
            line: Line::new_on_axis(p0.x,p0.y)
        }
    ).collect();
    

    let cov_model = VonKarmanLayer::new(0.15, 25.0, 0.0);
    
    let c_phi_phi = CovMat::new(&phi, &phi, &cov_model).matrix();
    Ok(c_phi_phi)
}

#[pyfunction]
fn build_dpc() -> PyResult<Vec<Vec<f64>>> {
    const NPHISAMPLES: u32 = 32;
    const NACTUX: u32 = 17;
    
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
    .map(|x| 
        yy.iter().map(move |y| {
            x+y
        })
    ).flatten().collect();
    
    let phi: Vec<Measurement> = phi_coords
    .iter()
    .map(|p0|
        Measurement::Phase{
            line: Line::new_on_axis(p0.x,p0.y)
        }
    ).collect();
    

    /////////////
    // define actuator related coordinates:
    let sf = NACTUX as f64 / (NACTUX-1) as f64;
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0*sf, 0.0),
        &Vec2D::new( 4.0*sf, 0.0),
        NACTUX,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0*sf),
        &Vec2D::new( 0.0,  4.0*sf),
        NACTUX,
    );
    let com_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let com: Vec<Actuator> = com_coords
    .iter()
    .map(move |p|
        Actuator::Gaussian{
            position: Vec3D::new(p.x, p.y, 0.0),
            sigma: coupling_to_sigma(0.3, 8.0/(NACTUX as f64 - 1.0)),
        }
    ).collect();


    let d_phi_com = IMat::new(&phi, &com).matrix();
    Ok(d_phi_com)
}



#[pyfunction]
fn build_cmm() -> PyResult<Vec<Vec<f64>>> {
    const AS2RAD: f64 = 4.848e-6;
    const NSUBX: u32 = 16;
    
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
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let wfs_dirs = vec![
        Vec2D::new(-10.0, -10.0),
        Vec2D::new(-10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
    ];
    let meas: Vec<Measurement> = wfs_dirs.into_iter()
    .map(|dir_arcsec|
        dir_arcsec * AS2RAD
    ).map(|dir|
        meas_coords
        .iter().map(move |p|
            vec![
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                },
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                }
            ]
        ).flatten()
    ).flatten().collect();

    let cov_model = VonKarmanLayer::new(0.15, 25.0, 0.0);
    
    let c_phip1_meas = CovMat::new(&meas, &meas, &cov_model).matrix();

    Ok(c_phip1_meas)
}



#[pyfunction]
fn build_cpm(delta_t: f64) -> PyResult<Vec<Vec<f64>>> {
    const AS2RAD: f64 = 4.848e-6;
    const NPHISAMPLES: u32 = 32;
    const NSUBX: u32 = 16;
    
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
    .map(|x| 
        yy.iter().map(move |y| {
            x+y
        })
    ).flatten().collect();
    
    let phip1: Vec<Measurement> = phi_coords
    .iter()
    .map(|p0|
        Measurement::Phase{
            line: Line::new_on_axis(p0.x+0.005*delta_t, p0.y)
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
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let wfs_dirs = vec![
        Vec2D::new(-10.0, -10.0),
        Vec2D::new(-10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
    ];
    let meas: Vec<Measurement> = wfs_dirs.into_iter()
    .map(|dir_arcsec|
        dir_arcsec * AS2RAD
    ).map(|dir|
        meas_coords
        .iter().map(move |p|
            vec![
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                },
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                }
            ]
        ).flatten()
    ).flatten().collect();

    let cov_model = VonKarmanLayer::new(0.15, 25.0, 0.0);
    
    let c_phip1_meas = CovMat::new(&phip1, &meas, &cov_model).matrix();

    Ok(c_phip1_meas)
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


#[pyfunction]
fn ultimatestart_system_matrices(verbose: bool) -> PyResult<InitMatrices> {
    const AS2RAD: f64 = 4.848e-6;
    const NPHISAMPLES: u32 = 64;
    const NSUBX: u32 = 32;
    const NACTUX: u32 = 64;
    
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
    .map(|x|
        yy.iter().map(move |y| {
            x+y
        })
    ).flatten().collect();
    
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
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let wfs_dirs = vec![
        Vec2D::new(-10.0, -10.0),
        Vec2D::new(-10.0,  10.0),
        Vec2D::new( 10.0, -10.0),
        Vec2D::new( 10.0,  10.0),
    ];
    let meas: Vec<Measurement> = wfs_dirs.into_iter()
    .map(|dir_arcsec|
        dir_arcsec * AS2RAD
    ).map(|dir|
        meas_coords
        .iter().map(move |p|
            vec![
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                },
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                }
            ]
        ).flatten()
    ).flatten().collect();

    /////////////
    // define actuator related coordinates:
    let sf = NACTUX as f64 / (NACTUX-1) as f64;
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0*sf, 0.0),
        &Vec2D::new( 4.0*sf, 0.0),
        NACTUX,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0*sf),
        &Vec2D::new( 0.0,  4.0*sf),
        NACTUX,
    );
    let com_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
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
    
    
    if verbose {
        println!("doing c_phi_phi matrix");
    }
    let c_phi_phi = CovMat::new(&phi, &phi, &cov_model).matrix();

    if verbose {
        println!("doing c_phip1_phi matrix");
    }
    let c_phip1_phi = CovMat::new(&phip1, &phi, &cov_model).matrix();

    if verbose {
        println!("doing c_meas_phi matrix");
    }
    let c_meas_phi = CovMat::new(&meas, &phi, &cov_model).matrix();

    if verbose {
        println!("doing d_meas_com matrix");
    }
    let d_meas_com = IMat::new(&meas, &com).matrix();

    if verbose {
        println!("doing d_phi_com matrix");
    }
    let d_phi_com = IMat::new(&phi, &com).matrix();

    let init_matrices = InitMatrices {
        c_phi_phi,
        c_phip1_phi,
        c_meas_phi,
        d_meas_com,
        d_phi_com,
    };
    Ok(init_matrices)
}

#[pyfunction]
fn build_system_matrices(verbose: bool) -> PyResult<InitMatrices> {
    const AS2RAD: f64 = 4.848e-6;
    const NPHISAMPLES: u32 = 32;
    const NSUBX: u32 = 16;
    const NACTUX: u32 = 17;
    
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
    .map(|x| 
        yy.iter().map(move |y| {
            x+y
        })
    ).flatten().collect();
    
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
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let wfs_dirs = vec![
        Vec2D::new(-10.0, -10.0),
        Vec2D::new(-10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
        Vec2D::new( 10.0,  10.0),
    ];
    let meas: Vec<Measurement> = wfs_dirs.into_iter()
    .map(|dir_arcsec|
        dir_arcsec * AS2RAD
    ).map(|dir|
        meas_coords
        .iter().map(move |p|
            vec![
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::x_unit(),
                    npoints: 2,
                },
                Measurement::SlopeTwoEdge{
                    central_line: Line::new(p.x, dir.x, p.y, dir.y),
                    edge_length: 0.25,
                    edge_separation: 0.25,
                    gradient_axis: Vec2D::y_unit(),
                    npoints: 2,
                }
            ]
        ).flatten()
    ).flatten().collect();

    /////////////
    // define actuator related coordinates:
    let sf = NACTUX as f64 / (NACTUX-1) as f64;
    let xx = Vec2D::linspace(
        &Vec2D::new(-4.0*sf, 0.0),
        &Vec2D::new( 4.0*sf, 0.0),
        NACTUX,
    );
    let yy = Vec2D::linspace(
        &Vec2D::new( 0.0, -4.0*sf),
        &Vec2D::new( 0.0,  4.0*sf),
        NACTUX,
    );
    let com_coords: Vec<Vec2D> = xx.iter()
    .map(|x| 
        yy.iter().map(|y| {
            x+y
        }).collect::<Vec<Vec2D>>()
    ).flatten().collect();
    let com: Vec<Actuator> = com_coords
    .iter()
    .map(move |p|
        Actuator::Gaussian{
            position: Vec3D::new(p.x, p.y, 0.0),
            sigma: coupling_to_sigma(0.3, 8.0/(NACTUX as f64 - 1.0)),
        }
    ).collect();


    let cov_model = VonKarmanLayer::new(0.15, 25.0, 0.0);
    
    
    if verbose {
        println!("doing c_phi_phi matrix");
    }
    let c_phi_phi = CovMat::new(&phi, &phi, &cov_model).matrix();

    if verbose {
        println!("doing c_phip1_phi matrix");
    }
    let c_phip1_phi = CovMat::new(&phip1, &phi, &cov_model).matrix();

    if verbose {
        println!("doing c_meas_phi matrix");
    }
    let c_meas_phi = CovMat::new(&meas, &phi, &cov_model).matrix();

    if verbose {
        println!("doing d_meas_com matrix");
    }
    let d_meas_com = IMat::new(&meas, &com).matrix();

    if verbose {
        println!("doing d_phi_com matrix");
    }
    let d_phi_com = IMat::new(&phi, &com).matrix();

    let init_matrices = InitMatrices {
        c_phi_phi,
        c_phip1_phi,
        c_meas_phi,
        d_meas_com,
        d_phi_com,
    };
    Ok(init_matrices)
}