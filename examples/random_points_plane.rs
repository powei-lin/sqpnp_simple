use std::time::Instant;

use nalgebra as na;
use rand::{self, random};
use sqpnp_simple::sqpnp_solve;

fn main() {
    env_logger::init();
    let p3ds_pt: Vec<_> = (0..144)
        .map(|_| na::Point3::<f64>::new(random(), random(), 0.5))
        .collect();

    let rvec = 3.0
        * na::Vector3::new(
            random::<f64>() - 0.5,
            random::<f64>() - 0.5,
            random::<f64>() - 0.5,
        );
    let tvec = na::Vector3::new(random::<f64>(), random::<f64>(), random::<f64>());
    let transform = na::Isometry3::new(tvec, rvec);
    let rvec = transform.rotation.scaled_axis();
    let (p2ds, p3ds): (Vec<(f64, f64)>, Vec<_>) = p3ds_pt
        .iter()
        .filter_map(|p| {
            let p3dt = transform * p;
            if p3dt.z > 0.0 {
                Some(((p3dt.x / p3dt.z, p3dt.y / p3dt.z), (p.x, p.y, p.z)))
            } else {
                None
            }
        })
        .unzip();
    let s = Instant::now();
    if let Some((rvec_pnp, tvec_pnp)) = sqpnp_solve(&p3ds, &p2ds) {
        let t = s.elapsed();
        let rvec_pnp = na::Vector3::new(rvec_pnp.0, rvec_pnp.1, rvec_pnp.2);
        let tvec_pnp = na::Vector3::new(tvec_pnp.0, tvec_pnp.1, tvec_pnp.2);
        println!("rvec gt  [{}, {}, {}]", rvec[0], rvec[1], rvec[2]);
        println!(
            "rvec pnp [{}, {}, {}]",
            rvec_pnp[0], rvec_pnp[1], rvec_pnp[2]
        );
        println!("tvec gt  [{}, {}, {}]", tvec[0], tvec[1], tvec[2]);
        println!(
            "tvec pnp [{}, {}, {}]",
            tvec_pnp[0], tvec_pnp[1], tvec_pnp[2]
        );
        println!("rvec diff {:?}", rvec_pnp - rvec);
        println!("tvec diff {:?}", tvec_pnp - tvec);
        println!("solving time: {} sec", t.as_secs_f64());

        let transform_pnp = na::Isometry3::new(tvec_pnp, rvec_pnp);
        println!("gt  rmat {}", transform.rotation.to_rotation_matrix());
        println!("pnp rmat {}", transform_pnp.rotation.to_rotation_matrix());
        // p3ds_pt[0]
    } else {
        println!("failed rnorm: {}", rvec.norm());
        println!("rvec gt  [{}, {}, {}]", rvec[0], rvec[1], rvec[2]);
        println!("tvec gt  [{}, {}, {}]", tvec[0], tvec[1], tvec[2]);
    }
}
