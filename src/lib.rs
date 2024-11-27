use std::{collections::HashMap, iter::zip};

use nalgebra::{self as na, Const};
use num_dual::DualDVec64;
use tiny_solver::{factors::Factor, Optimizer};

struct SQpnpFactor {
    omega: na::DMatrix<DualDVec64>,
}

impl SQpnpFactor {
    pub fn new(omega: na::DMatrix<f64>) -> SQpnpFactor {
        let omega = omega.cast();
        SQpnpFactor { omega }
    }
}

impl Factor for SQpnpFactor {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        let rvec = na::Vector3::new(
            params[0][0].clone(),
            params[0][1].clone(),
            params[0][2].clone(),
        );
        let r = na::Rotation3::from_scaled_axis(rvec.clone())
            .matrix()
            .clone_owned()
            .reshape_generic(Const::<9>, Const::<1>);
        let residual = r.transpose() * self.omega.clone() * r;
        na::dvector![
            residual[(0, 0)].clone(),
            residual[(0, 0)].clone(),
            residual[(0, 0)].clone()
        ]
    }
}

pub fn sqpnp_solve(p3ds: &[(f64, f64, f64)], p2ds_z: &[(f64, f64)]) -> () {
    if p3ds.len() < 3 {
        return;
    }
    if p3ds.len() != p2ds_z.len() {
        return;
    }

    let mut qmat_vec = Vec::new();
    let mut amat_vec = Vec::new();
    let mut q_sum = na::DMatrix::zeros(3, 3);
    let mut qa_sum = na::DMatrix::zeros(3, 9);

    for (&(p3x, p3y, p3z), &(p2x, p2y)) in zip(p3ds, p2ds_z) {
        let amat = na::dmatrix![p3x, p3y, p3z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0,p3x, p3y, p3z,  0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p3x, p3y, p3z];
        let q_mat = na::dmatrix![1.0, 0.0, -p2x; 0.0, 1.0, -p2y; -p2x, -p2y, p2x*p2x+p2y*p2y];
        q_sum += q_mat.clone();
        qa_sum += q_mat.clone() * amat.clone();
        amat_vec.push(amat);
        qmat_vec.push(q_mat);
    }
    let q_sum_inv_option = q_sum.try_inverse();
    if q_sum_inv_option.is_none() {
        return;
    }
    let pmat = -1.0 * q_sum_inv_option.unwrap() * qa_sum;
    // println!("p {:?}", pmat.shape());
    let mut omega = na::DMatrix::zeros(9, 9);
    for (amat, qmat) in zip(amat_vec, qmat_vec) {
        let a_plus_p = amat + pmat.clone();
        omega += a_plus_p.transpose() * qmat * a_plus_p;
    }

    let mut problem = tiny_solver::Problem::new();
    let cost = SQpnpFactor::new(omega);
    problem.add_residual_block(3, vec![("rvec".to_string(), 3)], Box::new(cost), None);

    let initial_values = HashMap::<String, na::DVector<f64>>::from([(
        "rvec".to_string(),
        na::dvector![0.0001, 0.01, 0.01],
    )]);

    // initialize optimizer
    let optimizer = tiny_solver::GaussNewtonOptimizer {};

    // optimize
    let result = optimizer.optimize(&problem, &initial_values, None);
    // println!("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
    }
}
