use core::f64;
use std::iter::zip;

use log::trace;
use nalgebra as na;

fn rvec_to_r9<T>(rvec: &na::Vector3<T>) -> na::SVector<T, 9>
where
    T: na::SimdRealField + na::RealField,
{
    // na reshape is row then col so we transpose first
    na::Rotation3::from_scaled_axis(rvec.clone())
        .matrix()
        .transpose()
        .clone_owned()
        .reshape_generic(na::Const::<9>, na::Const::<1>)
}

fn residual_and_approx_jaccobian(
    rvec: &na::Vector3<f64>,
    omegas: &[na::SMatrix<f64, 9, 9>],
) -> (na::DVector<f64>, na::DMatrix<f64>) {
    let mut jac = na::DMatrix::<f64>::zeros(omegas.len(), 3);
    let mut residuals = na::DVector::<f64>::zeros(omegas.len());
    const STEP: f64 = 1e-8;
    let r = rvec_to_r9(rvec);
    for (row, om) in omegas.iter().enumerate() {
        let residual = r.transpose() * om * r;
        residuals[row] = residual[0];
        for i in 0..3 {
            let mut rvec_plus = rvec.clone_owned();
            unsafe {
                *rvec_plus.get_unchecked_mut(i) += STEP;
            }
            let r_plus = rvec_to_r9(&rvec_plus);
            let residual_plus = r_plus.transpose() * om * r_plus;
            let j = (residual_plus[0] - residual[0]) / STEP;
            jac[(row, i)] = j;
        }
    }

    (residuals, jac)
}
pub fn sqpnp_solve_glam(
    p3ds: &[glam::Vec3],
    p2ds_z: &[glam::Vec2],
) -> Option<((f64, f64, f64), (f64, f64, f64))> {
    let p3ds: Vec<_> = p3ds
        .iter()
        .map(|p| (p.x as f64, p.y as f64, p.z as f64))
        .collect();
    let p2ds: Vec<_> = p2ds_z.iter().map(|p| (p.x as f64, p.y as f64)).collect();
    sqpnp_solve(&p3ds, &p2ds)
}

pub fn sqpnp_solve(
    p3ds: &[(f64, f64, f64)],
    p2ds_z: &[(f64, f64)],
) -> Option<((f64, f64, f64), (f64, f64, f64))> {
    const MAX_ITER: usize = 20;
    const MAX_OMAGA_SQUASH: usize = 6;
    const MAX_RVEC_STEP: f64 = 0.5;
    if p3ds.len() < 3 {
        return None;
    }
    if p3ds.len() != p2ds_z.len() {
        return None;
    }

    // prepare matrices
    let mut qmat_vec = Vec::new();
    let mut amat_vec = Vec::new();
    let mut q_sum = na::SMatrix::<f64, 3, 3>::zeros();
    let mut qa_sum = na::SMatrix::<f64, 3, 9>::zeros();

    for (&(p3x, p3y, p3z), &(p2x, p2y)) in zip(p3ds, p2ds_z) {
        let amat_data = vec![
            p3x, 0.0, 0.0, p3y, 0.0, 0.0, p3z, 0.0, 0.0, 0.0, p3x, 0.0, 0.0, p3y, 0.0, 0.0, p3z,
            0.0, 0.0, 0.0, p3x, 0.0, 0.0, p3y, 0.0, 0.0, p3z,
        ];
        let amat = na::SMatrix::<f64, 3, 9>::from_vec(amat_data);
        let q_mat_data = vec![
            1.0,
            0.0,
            -p2x,
            0.0,
            1.0,
            -p2y,
            -p2x,
            -p2y,
            p2x * p2x + p2y * p2y,
        ];
        let q_mat = na::SMatrix::<f64, 3, 3>::from_vec(q_mat_data);
        q_sum += q_mat;
        qa_sum += q_mat * amat;
        amat_vec.push(amat);
        qmat_vec.push(q_mat);
    }

    let q_sum_inv_option = q_sum.try_inverse();
    q_sum_inv_option?;

    let pmat = -1.0 * q_sum_inv_option.unwrap() * qa_sum;
    let num_omega = MAX_OMAGA_SQUASH.min(amat_vec.len());
    let mut omegas = vec![na::SMatrix::<f64, 9, 9>::zeros(); num_omega];
    for (i, (amat, qmat)) in zip(amat_vec, qmat_vec).enumerate() {
        let a_plus_p = amat + pmat;
        let omega = a_plus_p.transpose() * qmat * a_plus_p;
        omegas[i % num_omega] += omega;
    }

    let mut rvec = na::Vector3::<f64>::new(0.0, 0.0, 1e-4);
    let mut prev_dx_norm_squared = f64::MAX;
    for i in 0..MAX_ITER {
        let (residuals, jac) = residual_and_approx_jaccobian(&rvec, &omegas);
        let b = -1.0 * jac.transpose() * residuals;
        let a = jac.transpose() * jac;

        let mut dx = a.qr().solve(&b).unwrap();
        dx.iter_mut().for_each(|i| {
            if i.abs() > MAX_RVEC_STEP {
                *i /= i.abs() / MAX_RVEC_STEP;
            }
        });
        let ns = dx.norm_squared();
        trace!("{} dx {}", i, ns);
        rvec += dx;
        if ns < 1e-10 {
            break;
        } else if ns > prev_dx_norm_squared {
            trace!("dx not decreasing reset.");
            rvec = na::Rotation3::from_scaled_axis(na::Vector3::new_random()).scaled_axis();
            prev_dx_norm_squared = f64::MAX;
        } else if rvec.norm() > f64::consts::PI {
            trace!("rvec norm larger than pi, reset.");
            rvec = na::Rotation3::from_scaled_axis(na::Vector3::new_random()).scaled_axis();
        } else {
            prev_dx_norm_squared = ns;
        }
    }
    if prev_dx_norm_squared > 0.0001 {
        return None;
    }
    let tvec = pmat * rvec_to_r9(&rvec);
    let transform = na::Isometry3::new(tvec, rvec);
    let p3d0 = na::Point3::new(p3ds[0].0, p3ds[0].1, p3ds[0].2);
    let p3dz = (transform * p3d0).z;
    if p3dz > 0.0 {
        Some(((rvec[0], rvec[1], rvec[2]), (tvec[0], tvec[1], tvec[2])))
    } else {
        let mut rmat = na::Rotation3::from_scaled_axis(rvec).matrix().clone();
        trace!("rmat before {}", rmat);
        rmat.columns_mut(0, 2).scale_mut(-1.0);
        let rvec = na::Rotation3::from_matrix_unchecked(rmat).scaled_axis();
        let tvec = pmat * rvec_to_r9(&rvec);

        Some(((rvec[0], rvec[1], rvec[2]), (tvec[0], tvec[1], tvec[2])))
    }
}
