use std::time::Instant;

use nalgebra as na;
use rand::{self, random};
use sqpnp_simple::sqpnp_solve;

fn main() {
    env_logger::init();
    let p3ds_pt: Vec<_> = (0..144)
        .map(|_| na::Point3::<f64>::new(random(), random(), random::<f64>() + 0.5))
        .collect();

    let rvec = na::Vector3::new(
        random::<f64>() - 0.5,
        random::<f64>() - 0.5,
        random::<f64>() - 0.5,
    );
    let tvec = na::Vector3::new(random::<f64>(), random::<f64>(), random::<f64>());
    let transform = na::Isometry3::new(tvec, rvec);
    let p2ds: Vec<(f64, f64)> = p3ds_pt
        .iter()
        .map(|p| {
            let p3dt = transform * p;
            (p3dt.x / p3dt.z, p3dt.y / p3dt.z)
        })
        .collect();
    let p3ds: Vec<_> = p3ds_pt.iter().map(|p| (p.x, p.y, p.z)).collect();
    let s = Instant::now();
    if let Some((rvec_pnp, tvec_pnp)) = sqpnp_solve(&p3ds, &p2ds) {
        let t = s.elapsed();
        let rvec_pnp = na::Vector3::new(rvec_pnp.0, rvec_pnp.1, rvec_pnp.2);
        let tvec_pnp = na::Vector3::new(tvec_pnp.0, tvec_pnp.1, tvec_pnp.2);
        println!("rvec gt  {}", rvec);
        println!("rvec pnp {}", rvec_pnp);
        println!("tvec gt  {}", tvec);
        println!("tvec pnp {}", tvec_pnp);
        println!("rvec diff {:?}", rvec_pnp - rvec);
        println!("tvec diff {:?}", tvec_pnp - tvec);
        println!("solving time: {} sec", t.as_secs_f64());
    };
}
