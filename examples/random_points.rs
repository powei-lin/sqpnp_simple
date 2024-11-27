use nalgebra as na;
use rand::{self, random};
use sqpnp_simple::sqpnp_solve;

fn main() {
    let p3ds_pt: Vec<_> = (0..100)
        .into_iter()
        .map(|_| na::Point3::<f64>::new(random(), random(), random::<f64>() + 0.5))
        .collect();

    let rvec = na::Vector3::new(random::<f64>(), random::<f64>(), random::<f64>());
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

    sqpnp_solve(&p3ds, &p2ds);
}
