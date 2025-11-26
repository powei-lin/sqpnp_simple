use diol::prelude::*;
use nalgebra as na;
use rand::random;
use sqpnp_simple::sqpnp_solve;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::default());

    let args: Vec<(String, usize)> = vec![
        ("random".to_string(), 10),
        ("random".to_string(), 100),
        ("random".to_string(), 1000),
        ("planar".to_string(), 10),
        ("planar".to_string(), 100),
        ("planar".to_string(), 1000),
    ];

    bench.register(
        |b: Bencher, (scenario, n): (String, usize)| {
            let n = n;
            let (p3ds, p2ds) = if scenario == "random" {
                let p3ds_pt: Vec<_> = (0..n)
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
                (p3ds, p2ds)
            } else {
                // Planar case
                let p3ds_pt: Vec<_> = (0..n)
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
                (p3ds, p2ds)
            };

            b.bench(|| sqpnp_solve(&p3ds, &p2ds))
        },
        args,
    );

    bench.run()?;
    Ok(())
}
