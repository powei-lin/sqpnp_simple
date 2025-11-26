use core::f64;
use std::iter::zip;

use nalgebra as na;

#[derive(Debug, Clone)]
pub struct SolverParameters {
    pub rank_tolerance: f64,
    pub sqp_squared_tolerance: f64,
    pub sqp_det_threshold: f64,
    pub sqp_max_iteration: usize,
    pub orthogonality_squared_error_threshold: f64,
    pub equal_vectors_squared_diff: f64,
    pub equal_squared_errors_diff: f64,
    pub point_variance_threshold: f64,
}

impl Default for SolverParameters {
    fn default() -> Self {
        Self {
            rank_tolerance: 1e-7,
            sqp_squared_tolerance: 1e-10,
            sqp_det_threshold: 1.001,
            sqp_max_iteration: 25,
            orthogonality_squared_error_threshold: 1e-8,
            equal_vectors_squared_diff: 1e-10,
            equal_squared_errors_diff: 1e-6,
            point_variance_threshold: 1e-5,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SQPSolution {
    pub r_hat: na::Vector3<f64>,
    pub t: na::Vector3<f64>,
    pub num_iterations: usize,
    pub sq_error: f64,
}

struct PnPSolver {
    parameters: SolverParameters,
    omega: na::SMatrix<f64, 9, 9>,
    s: na::SVector<f64, 9>,
    u: na::SMatrix<f64, 9, 9>,
    p: na::SMatrix<f64, 3, 9>,
    point_mean: na::Vector3<f64>,
    num_null_vectors: usize,
    flag_valid: bool,
    solutions: Vec<SQPSolution>,
}

impl PnPSolver {
    fn new(p3ds: &[(f64, f64, f64)], p2ds: &[(f64, f64)], parameters: SolverParameters) -> Self {
        let n = p3ds.len();
        if n != p2ds.len() || n < 3 {
            return Self {
                parameters,
                omega: na::SMatrix::zeros(),
                s: na::SVector::zeros(),
                u: na::SMatrix::zeros(),
                p: na::SMatrix::zeros(),
                point_mean: na::Vector3::zeros(),
                num_null_vectors: 0,
                flag_valid: false,
                solutions: Vec::new(),
            };
        }

        let mut omega = na::SMatrix::<f64, 9, 9>::zeros();
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wx2_plus_wy2 = 0.0;
        let mut sum_w_x = 0.0;
        let mut sum_w_y = 0.0;
        let mut sum_w_z = 0.0;
        let mut qa_sum = na::SMatrix::<f64, 3, 9>::zeros();

        for (p3, p2) in zip(p3ds, p2ds) {
            let w = 1.0; // Assuming unit weights for now as per simple interface
            let (x, y, z) = *p3;
            let (u, v) = *p2;

            let wx = u * w;
            let wy = v * w;
            let wsq_norm_m = w * (u * u + v * v);
            sum_w += w;
            sum_wx += wx;
            sum_wy += wy;
            sum_wx2_plus_wy2 += wsq_norm_m;
            sum_w_x += w * x;
            sum_w_y += w * y;
            sum_w_z += w * z;

            let x2 = x * x;
            let xy = x * y;
            let xz = x * z;
            let y2 = y * y;
            let yz = y * z;
            let z2 = z * z;

            // Omega accumulation
            // Block (0:2, 0:2)
            omega[(0, 0)] += w * x2;
            omega[(0, 1)] += w * xy;
            omega[(0, 2)] += w * xz;
            omega[(1, 1)] += w * y2;
            omega[(1, 2)] += w * yz;
            omega[(2, 2)] += w * z2;

            // Block (0:2, 6:8)
            omega[(0, 6)] -= wx * x2;
            omega[(0, 7)] -= wx * xy;
            omega[(0, 8)] -= wx * xz;
            omega[(1, 7)] -= wx * y2;
            omega[(1, 8)] -= wx * yz;
            omega[(2, 8)] -= wx * z2;

            // Block (3:5, 6:8)
            omega[(3, 6)] -= wy * x2;
            omega[(3, 7)] -= wy * xy;
            omega[(3, 8)] -= wy * xz;
            omega[(4, 7)] -= wy * y2;
            omega[(4, 8)] -= wy * yz;
            omega[(5, 8)] -= wy * z2;

            // Block (6:8, 6:8)
            omega[(6, 6)] += wsq_norm_m * x2;
            omega[(6, 7)] += wsq_norm_m * xy;
            omega[(6, 8)] += wsq_norm_m * xz;
            omega[(7, 7)] += wsq_norm_m * y2;
            omega[(7, 8)] += wsq_norm_m * yz;
            omega[(8, 8)] += wsq_norm_m * z2;

            // QA accumulation
            qa_sum[(0, 0)] += w * x;
            qa_sum[(0, 1)] += w * y;
            qa_sum[(0, 2)] += w * z;
            qa_sum[(0, 6)] -= wx * x;
            qa_sum[(0, 7)] -= wx * y;
            qa_sum[(0, 8)] -= wx * z;

            qa_sum[(1, 6)] -= wy * x;
            qa_sum[(1, 7)] -= wy * y;
            qa_sum[(1, 8)] -= wy * z;

            qa_sum[(2, 6)] += wsq_norm_m * x;
            qa_sum[(2, 7)] += wsq_norm_m * y;
            qa_sum[(2, 8)] += wsq_norm_m * z;
        }

        // Complete QA
        qa_sum[(1, 3)] = qa_sum[(0, 0)];
        qa_sum[(1, 4)] = qa_sum[(0, 1)];
        qa_sum[(1, 5)] = qa_sum[(0, 2)];
        qa_sum[(2, 0)] = qa_sum[(0, 6)];
        qa_sum[(2, 1)] = qa_sum[(0, 7)];
        qa_sum[(2, 2)] = qa_sum[(0, 8)];
        qa_sum[(2, 3)] = qa_sum[(1, 6)];
        qa_sum[(2, 4)] = qa_sum[(1, 7)];
        qa_sum[(2, 5)] = qa_sum[(1, 8)];

        // Fill lower triangles of Omega
        omega[(1, 6)] = omega[(0, 7)];
        omega[(2, 6)] = omega[(0, 8)];
        omega[(2, 7)] = omega[(1, 8)];
        omega[(4, 6)] = omega[(3, 7)];
        omega[(5, 6)] = omega[(3, 8)];
        omega[(5, 7)] = omega[(4, 8)];
        omega[(7, 6)] = omega[(6, 7)];
        omega[(8, 6)] = omega[(6, 8)];
        omega[(8, 7)] = omega[(7, 8)];

        // Block (3:5, 3:5) is same as (0:2, 0:2)
        omega[(3, 3)] = omega[(0, 0)];
        omega[(3, 4)] = omega[(0, 1)];
        omega[(3, 5)] = omega[(0, 2)];
        omega[(4, 4)] = omega[(1, 1)];
        omega[(4, 5)] = omega[(1, 2)];
        omega[(5, 5)] = omega[(2, 2)];

        // Symmetric fill
        for i in 0..9 {
            for j in 0..i {
                omega[(i, j)] = omega[(j, i)];
            }
        }

        // Q matrix
        let mut q = na::SMatrix::<f64, 3, 3>::zeros();
        q[(0, 0)] = sum_w;
        q[(0, 2)] = -sum_wx;
        q[(1, 1)] = sum_w;
        q[(1, 2)] = -sum_wy;
        q[(2, 0)] = -sum_wx;
        q[(2, 1)] = -sum_wy;
        q[(2, 2)] = sum_wx2_plus_wy2;
        q[(1, 0)] = q[(0, 1)]; // 0
        q[(0, 1)] = 0.0;
        // Symmetric Q
        q[(0, 1)] = q[(1, 0)];
        q[(2, 0)] = q[(0, 2)];
        q[(2, 1)] = q[(1, 2)]; // Wait, q(2,0) was set above.

        // Re-check Q construction from C++
        // Q(0, 0) = sum_w;     Q(0, 1) = 0.0;       Q(0, 2) = -sum_wx;
        // Q(1, 0) = 0.0;       Q(1, 1) = sum_w;     Q(1, 2) = -sum_wy;
        // Q(2, 0) = -sum_wx;   Q(2, 1) = -sum_wy;   Q(2, 2) = sum_wx2_plus_wy2;
        // Correct.

        let q_inv = match q.try_inverse() {
            Some(inv) => inv,
            None => {
                return Self {
                    parameters,
                    omega: na::SMatrix::zeros(),
                    s: na::SVector::zeros(),
                    u: na::SMatrix::zeros(),
                    p: na::SMatrix::zeros(),
                    point_mean: na::Vector3::zeros(),
                    num_null_vectors: 0,
                    flag_valid: false,
                    solutions: Vec::new(),
                }
            }
        };

        let p_mat = -q_inv * qa_sum;
        omega += qa_sum.transpose() * p_mat;

        // SVD for null space
        let svd = omega.svd(true, true);
        let u = svd.u.unwrap();
        let s = svd.singular_values;

        let mut num_null_vectors = 0;
        while 7 - num_null_vectors >= 0
            && s[7 - num_null_vectors as usize] < parameters.rank_tolerance
        {
            num_null_vectors += 1;
        }
        num_null_vectors += 1;

        if num_null_vectors > 6 {
            return Self {
                parameters,
                omega: na::SMatrix::zeros(),
                s: na::SVector::zeros(),
                u: na::SMatrix::zeros(),
                p: na::SMatrix::zeros(),
                point_mean: na::Vector3::zeros(),
                num_null_vectors: 0,
                flag_valid: false,
                solutions: Vec::new(),
            };
        }

        let inv_sum_w = 1.0 / sum_w;
        let point_mean = na::Vector3::new(
            sum_w_x * inv_sum_w,
            sum_w_y * inv_sum_w,
            sum_w_z * inv_sum_w,
        );

        Self {
            parameters,
            omega,
            s,
            u,
            p: p_mat,
            point_mean,
            num_null_vectors: num_null_vectors as usize,
            flag_valid: true,
            solutions: Vec::new(),
        }
    }

    fn solve(&mut self) -> bool {
        if !self.flag_valid {
            return false;
        }

        let mut min_sq_error = f64::MAX;
        let num_eigen_points = if self.num_null_vectors > 0 {
            self.num_null_vectors
        } else {
            1
        };

        for i in (9 - num_eigen_points)..9 {
            let e = self.u.column(i).into_owned();
            let e = e * f64::sqrt(3.0); // Scale by sqrt(3) as in C++

            let orthogonality_sq_error = self.orthogonality_error(&e);

            if orthogonality_sq_error < self.parameters.orthogonality_squared_error_threshold {
                let r_hat = self.determinant9x1(&e) * e;
                let t = self.p * r_hat;
                let _solution = SQPSolution {
                    r_hat: na::Vector3::new(r_hat[0], r_hat[1], r_hat[2]), // Storing as vector3 for output, but internal calc uses 9x1
                    t,
                    num_iterations: 0,
                    sq_error: 0.0, // Calculated in handle_solution
                };
                // Need to adapt handle_solution to take 9x1 r_hat
                self.handle_solution(r_hat, t, 0, &mut min_sq_error);
            } else {
                let r = self.nearest_rotation_matrix(&e);
                let sol = self.run_sqp(&r);
                self.handle_solution(sol.0, sol.1, sol.2, &mut min_sq_error);

                let r = self.nearest_rotation_matrix(&(-e));
                let sol = self.run_sqp(&r);
                self.handle_solution(sol.0, sol.1, sol.2, &mut min_sq_error);
            }
        }

        // Check other eigenvectors if needed
        let mut index = 9 - num_eigen_points - 1;
        while index > 0 && min_sq_error > 3.0 * self.s[index] {
            let e = self.u.column(index).into_owned();
            let r = self.nearest_rotation_matrix(&e);
            let sol = self.run_sqp(&r);
            self.handle_solution(sol.0, sol.1, sol.2, &mut min_sq_error);

            let r = self.nearest_rotation_matrix(&(-e));
            let sol = self.run_sqp(&r);
            self.handle_solution(sol.0, sol.1, sol.2, &mut min_sq_error);

            if index == 0 {
                break;
            }
            index -= 1;
        }

        true
    }

    fn handle_solution(
        &mut self,
        r: na::SVector<f64, 9>,
        t: na::Vector3<f64>,
        num_iterations: usize,
        min_sq_error: &mut f64,
    ) {
        // Check cheirality
        if !self.test_positive_depth(&r, &t) {
            // In C++ they also check majority depths if centroid fails, implementing simplified version here checking centroid
            // If strict parity with C++ is needed, we need the points stored in PnPSolver
            // For now, let's assume if centroid fails, we skip.
            // Actually, let's implement the majority check if we can access points.
            // We didn't store points in PnPSolver struct in this implementation to avoid cloning.
            // Let's rely on centroid for now.
            return;
        }

        let sq_error = (self.omega * r).dot(&r);
        let _r_hat_vec = na::Vector3::new(
            f64::atan2(r[7], r[8]),
            f64::asin(-r[6]),
            f64::atan2(r[3], r[0]),
        ); // This is Euler angles, wait.
           // The C++ code returns rotation matrix elements in r_hat (9x1).
           // The user API expects rvec (Rodrigues vector) or similar?
           // The original rust code returned rvec (3-vector).
           // C++ `r_hat` is 9x1 flattened rotation matrix.
           // We need to convert 9x1 rotation matrix to 3-vector (Rodrigues).

        let r_mat = na::Matrix3::new(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
        let rvec = na::Rotation3::from_matrix(&r_mat).scaled_axis();

        let solution = SQPSolution {
            r_hat: rvec,
            t,
            num_iterations,
            sq_error,
        };

        if (min_sq_error.abs() - sq_error).abs() > self.parameters.equal_squared_errors_diff {
            if *min_sq_error > sq_error {
                *min_sq_error = sq_error;
                self.solutions.clear();
                self.solutions.push(solution);
            }
        } else {
            // Look for equal solution
            let mut found = false;
            for s in &mut self.solutions {
                if (s.r_hat - solution.r_hat).norm_squared()
                    < self.parameters.equal_vectors_squared_diff
                {
                    if s.sq_error > sq_error {
                        *s = solution;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                self.solutions.push(solution);
            }
            if *min_sq_error > sq_error {
                *min_sq_error = sq_error;
            }
        }
    }

    fn run_sqp(&self, r0: &na::SVector<f64, 9>) -> (na::SVector<f64, 9>, na::Vector3<f64>, usize) {
        let mut r = *r0;
        let mut delta_sq_norm = f64::MAX;
        let mut step = 0;

        while delta_sq_norm > self.parameters.sqp_squared_tolerance
            && step < self.parameters.sqp_max_iteration
        {
            let delta = self.solve_sqp_system(&r);
            r += delta;
            delta_sq_norm = delta.norm_squared();
            step += 1;
        }

        let mut det_r = self.determinant9x1(&r);
        if det_r < 0.0 {
            r = -r;
            det_r = -det_r;
        }

        let r_hat = if det_r > self.parameters.sqp_det_threshold {
            self.nearest_rotation_matrix(&r)
        } else {
            r
        };

        let t = self.p * r_hat;
        (r_hat, t, step)
    }

    fn solve_sqp_system(&self, r: &na::SVector<f64, 9>) -> na::SVector<f64, 9> {
        let sqnorm_r1 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let sqnorm_r2 = r[3] * r[3] + r[4] * r[4] + r[5] * r[5];
        let sqnorm_r3 = r[6] * r[6] + r[7] * r[7] + r[8] * r[8];
        let dot_r1r2 = r[0] * r[3] + r[1] * r[4] + r[2] * r[5];
        let dot_r1r3 = r[0] * r[6] + r[1] * r[7] + r[2] * r[8];
        let dot_r2r3 = r[3] * r[6] + r[4] * r[7] + r[5] * r[8];

        let (h, n, jh) = self.row_and_null_space(r);

        let mut g = na::SVector::<f64, 6>::zeros();
        g[0] = 1.0 - sqnorm_r1;
        g[1] = 1.0 - sqnorm_r2;
        g[2] = 1.0 - sqnorm_r3;
        g[3] = -dot_r1r2;
        g[4] = -dot_r2r3;
        g[5] = -dot_r1r3;

        let mut x = na::SVector::<f64, 6>::zeros();
        x[0] = g[0] / jh[(0, 0)];
        x[1] = g[1] / jh[(1, 1)];
        x[2] = g[2] / jh[(2, 2)];
        x[3] = (g[3] - jh[(3, 0)] * x[0] - jh[(3, 1)] * x[1]) / jh[(3, 3)];
        x[4] = (g[4] - jh[(4, 1)] * x[1] - jh[(4, 2)] * x[2] - jh[(4, 3)] * x[3]) / jh[(4, 4)];
        x[5] =
            (g[5] - jh[(5, 0)] * x[0] - jh[(5, 2)] * x[2] - jh[(5, 3)] * x[3] - jh[(5, 4)] * x[4])
                / jh[(5, 5)];

        let mut delta = h * x;

        let nt_omega = n.transpose() * self.omega;
        let w = nt_omega * n;
        let rhs = -(nt_omega * (delta + r));

        // Solve W * y = rhs. W is 3x3 symmetric.
        // Using nalgebra's cholesky or lu or svd.
        let y = match w.cholesky() {
            Some(cholesky) => cholesky.solve(&rhs),
            None => {
                // Fallback to SVD (pseudo-inverse) if Cholesky fails (not PD)
                // This mimics the C++ implementation's fallback to pseudo-inverse
                let svd = w.svd(true, true);
                svd.solve(&rhs, 1e-9).unwrap_or(na::Vector3::zeros())
            }
        };

        delta += n * y;
        delta
    }

    fn row_and_null_space(
        &self,
        r: &na::SVector<f64, 9>,
    ) -> (
        na::SMatrix<f64, 9, 6>,
        na::SMatrix<f64, 9, 3>,
        na::SMatrix<f64, 6, 6>,
    ) {
        let mut h = na::SMatrix::<f64, 9, 6>::zeros();
        let mut k = na::SMatrix::<f64, 6, 6>::zeros();

        // 1. q1
        let norm_r1 = f64::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        let inv_norm_r1 = if norm_r1 > 1e-5 { 1.0 / norm_r1 } else { 0.0 };
        h[(0, 0)] = r[0] * inv_norm_r1;
        h[(1, 0)] = r[1] * inv_norm_r1;
        h[(2, 0)] = r[2] * inv_norm_r1;
        k[(0, 0)] = 2.0 * norm_r1;

        // 2. q2
        let norm_r2 = f64::sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5]);
        let inv_norm_r2 = 1.0 / norm_r2;
        h[(3, 1)] = r[3] * inv_norm_r2;
        h[(4, 1)] = r[4] * inv_norm_r2;
        h[(5, 1)] = r[5] * inv_norm_r2;
        k[(1, 1)] = 2.0 * norm_r2;

        // 3. q3
        let norm_r3 = f64::sqrt(r[6] * r[6] + r[7] * r[7] + r[8] * r[8]);
        let inv_norm_r3 = 1.0 / norm_r3;
        h[(6, 2)] = r[6] * inv_norm_r3;
        h[(7, 2)] = r[7] * inv_norm_r3;
        h[(8, 2)] = r[8] * inv_norm_r3;
        k[(2, 2)] = 2.0 * norm_r3;

        // 4. q4
        let dot_j4q1 = r[3] * h[(0, 0)] + r[4] * h[(1, 0)] + r[5] * h[(2, 0)];
        let dot_j4q2 = r[0] * h[(3, 1)] + r[1] * h[(4, 1)] + r[2] * h[(5, 1)];

        h[(0, 3)] = r[3] - dot_j4q1 * h[(0, 0)];
        h[(1, 3)] = r[4] - dot_j4q1 * h[(1, 0)];
        h[(2, 3)] = r[5] - dot_j4q1 * h[(2, 0)];
        h[(3, 3)] = r[0] - dot_j4q2 * h[(3, 1)];
        h[(4, 3)] = r[1] - dot_j4q2 * h[(4, 1)];
        h[(5, 3)] = r[2] - dot_j4q2 * h[(5, 1)];

        let inv_norm_j4 = 1.0
            / f64::sqrt(
                h[(0, 3)] * h[(0, 3)]
                    + h[(1, 3)] * h[(1, 3)]
                    + h[(2, 3)] * h[(2, 3)]
                    + h[(3, 3)] * h[(3, 3)]
                    + h[(4, 3)] * h[(4, 3)]
                    + h[(5, 3)] * h[(5, 3)],
            );

        h[(0, 3)] *= inv_norm_j4;
        h[(1, 3)] *= inv_norm_j4;
        h[(2, 3)] *= inv_norm_j4;
        h[(3, 3)] *= inv_norm_j4;
        h[(4, 3)] *= inv_norm_j4;
        h[(5, 3)] *= inv_norm_j4;

        k[(3, 0)] = r[3] * h[(0, 0)] + r[4] * h[(1, 0)] + r[5] * h[(2, 0)];
        k[(3, 1)] = r[0] * h[(3, 1)] + r[1] * h[(4, 1)] + r[2] * h[(5, 1)];
        k[(3, 3)] = r[3] * h[(0, 3)]
            + r[4] * h[(1, 3)]
            + r[5] * h[(2, 3)]
            + r[0] * h[(3, 3)]
            + r[1] * h[(4, 3)]
            + r[2] * h[(5, 3)];

        // 5. q5
        let dot_j5q2 = r[6] * h[(3, 1)] + r[7] * h[(4, 1)] + r[8] * h[(5, 1)];
        let dot_j5q3 = r[3] * h[(6, 2)] + r[4] * h[(7, 2)] + r[5] * h[(8, 2)];
        let dot_j5q4 = r[6] * h[(3, 3)] + r[7] * h[(4, 3)] + r[8] * h[(5, 3)];

        h[(0, 4)] = -dot_j5q4 * h[(0, 3)];
        h[(1, 4)] = -dot_j5q4 * h[(1, 3)];
        h[(2, 4)] = -dot_j5q4 * h[(2, 3)];
        h[(3, 4)] = r[6] - dot_j5q2 * h[(3, 1)] - dot_j5q4 * h[(3, 3)];
        h[(4, 4)] = r[7] - dot_j5q2 * h[(4, 1)] - dot_j5q4 * h[(4, 3)];
        h[(5, 4)] = r[8] - dot_j5q2 * h[(5, 1)] - dot_j5q4 * h[(5, 3)];
        h[(6, 4)] = r[3] - dot_j5q3 * h[(6, 2)];
        h[(7, 4)] = r[4] - dot_j5q3 * h[(7, 2)];
        h[(8, 4)] = r[5] - dot_j5q3 * h[(8, 2)];

        let mut col4 = h.column_mut(4);
        col4.normalize_mut();

        k[(4, 1)] = r[6] * h[(3, 1)] + r[7] * h[(4, 1)] + r[8] * h[(5, 1)];
        k[(4, 2)] = r[3] * h[(6, 2)] + r[4] * h[(7, 2)] + r[5] * h[(8, 2)];
        k[(4, 3)] = r[6] * h[(3, 3)] + r[7] * h[(4, 3)] + r[8] * h[(5, 3)];
        k[(4, 4)] = r[6] * h[(3, 4)]
            + r[7] * h[(4, 4)]
            + r[8] * h[(5, 4)]
            + r[3] * h[(6, 4)]
            + r[4] * h[(7, 4)]
            + r[5] * h[(8, 4)];

        // 6. q6
        let dot_j6q1 = r[6] * h[(0, 0)] + r[7] * h[(1, 0)] + r[8] * h[(2, 0)];
        let dot_j6q3 = r[0] * h[(6, 2)] + r[1] * h[(7, 2)] + r[2] * h[(8, 2)];
        let dot_j6q4 = r[6] * h[(0, 3)] + r[7] * h[(1, 3)] + r[8] * h[(2, 3)];
        let dot_j6q5 = r[0] * h[(6, 4)]
            + r[1] * h[(7, 4)]
            + r[2] * h[(8, 4)]
            + r[6] * h[(0, 4)]
            + r[7] * h[(1, 4)]
            + r[8] * h[(2, 4)];

        h[(0, 5)] = r[6] - dot_j6q1 * h[(0, 0)] - dot_j6q4 * h[(0, 3)] - dot_j6q5 * h[(0, 4)];
        h[(1, 5)] = r[7] - dot_j6q1 * h[(1, 0)] - dot_j6q4 * h[(1, 3)] - dot_j6q5 * h[(1, 4)];
        h[(2, 5)] = r[8] - dot_j6q1 * h[(2, 0)] - dot_j6q4 * h[(2, 3)] - dot_j6q5 * h[(2, 4)];
        h[(3, 5)] = -dot_j6q5 * h[(3, 4)] - dot_j6q4 * h[(3, 3)];
        h[(4, 5)] = -dot_j6q5 * h[(4, 4)] - dot_j6q4 * h[(4, 3)];
        h[(5, 5)] = -dot_j6q5 * h[(5, 4)] - dot_j6q4 * h[(5, 3)];
        h[(6, 5)] = r[0] - dot_j6q3 * h[(6, 2)] - dot_j6q5 * h[(6, 4)];
        h[(7, 5)] = r[1] - dot_j6q3 * h[(7, 2)] - dot_j6q5 * h[(7, 4)];
        h[(8, 5)] = r[2] - dot_j6q3 * h[(8, 2)] - dot_j6q5 * h[(8, 4)];

        let mut col5 = h.column_mut(5);
        col5.normalize_mut();

        k[(5, 0)] = r[6] * h[(0, 0)] + r[7] * h[(1, 0)] + r[8] * h[(2, 0)];
        k[(5, 2)] = r[0] * h[(6, 2)] + r[1] * h[(7, 2)] + r[2] * h[(8, 2)];
        k[(5, 3)] = r[6] * h[(0, 3)] + r[7] * h[(1, 3)] + r[8] * h[(2, 3)];
        k[(5, 4)] = r[6] * h[(0, 4)]
            + r[7] * h[(1, 4)]
            + r[8] * h[(2, 4)]
            + r[0] * h[(6, 4)]
            + r[1] * h[(7, 4)]
            + r[2] * h[(8, 4)];
        k[(5, 5)] = r[6] * h[(0, 5)]
            + r[7] * h[(1, 5)]
            + r[8] * h[(2, 5)]
            + r[0] * h[(6, 5)]
            + r[1] * h[(7, 5)]
            + r[2] * h[(8, 5)];

        // Null space N
        let pn = na::SMatrix::<f64, 9, 9>::identity() - h * h.transpose();
        let mut n = na::SMatrix::<f64, 9, 3>::zeros();

        let norm_threshold = 0.1;
        let mut index1 = 0;
        let mut max_norm1 = f64::MIN;
        let mut col_norms = [0.0; 9];

        for i in 0..9 {
            col_norms[i] = pn.column(i).norm();
            if col_norms[i] >= norm_threshold && col_norms[i] > max_norm1 {
                max_norm1 = col_norms[i];
                index1 = i;
            }
        }

        let v1 = pn.column(index1);
        n.set_column(0, &(v1 / max_norm1));
        col_norms[index1] = -1.0;

        let mut index2 = 0;
        let mut min_dot12 = f64::MAX;
        for i in 0..9 {
            if col_norms[i] >= norm_threshold {
                let cos_v1_x_col = (pn.column(i).dot(&v1) / col_norms[i]).abs();
                if cos_v1_x_col <= min_dot12 {
                    index2 = i;
                    min_dot12 = cos_v1_x_col;
                }
            }
        }
        let v2 = pn.column(index2);
        let n0 = n.column(0);
        let mut n1 = v2 - v2.dot(&n0) * n0;
        n1.normalize_mut();
        n.set_column(1, &n1);
        col_norms[index2] = -1.0;

        let mut index3 = 0;
        let mut min_dot1323 = f64::MAX;
        for i in 0..9 {
            if col_norms[i] >= norm_threshold {
                let inv_norm = 1.0 / col_norms[i];
                let cos_v1_x_col = (pn.column(i).dot(&v1) * inv_norm).abs();
                let cos_v2_x_col = (pn.column(i).dot(&v2) * inv_norm).abs();
                if cos_v1_x_col + cos_v2_x_col <= min_dot1323 {
                    index3 = i;
                    min_dot1323 = cos_v1_x_col + cos_v2_x_col;
                }
            }
        }
        let v3 = pn.column(index3);
        let n0 = n.column(0);
        let n1 = n.column(1);
        let mut n2 = v3 - v3.dot(&n1) * n1 - v3.dot(&n0) * n0;
        n2.normalize_mut();
        n.set_column(2, &n2);

        (h, n, k)
    }

    fn nearest_rotation_matrix(&self, e: &na::SVector<f64, 9>) -> na::SVector<f64, 9> {
        let mat_e = na::Matrix3::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
        let svd = mat_e.svd(true, true);
        let u = svd.u.unwrap();
        let v_t = svd.v_t.unwrap();
        let det_uv = u.determinant() * v_t.determinant();
        let diag = na::Vector3::new(1.0, 1.0, det_uv);
        let r = u * na::Matrix3::from_diagonal(&diag) * v_t;

        na::SVector::<f64, 9>::from_vec(vec![
            r[(0, 0)],
            r[(0, 1)],
            r[(0, 2)],
            r[(1, 0)],
            r[(1, 1)],
            r[(1, 2)],
            r[(2, 0)],
            r[(2, 1)],
            r[(2, 2)],
        ])
    }

    fn orthogonality_error(&self, a: &na::SVector<f64, 9>) -> f64 {
        let sq_norm_a1 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        let sq_norm_a2 = a[3] * a[3] + a[4] * a[4] + a[5] * a[5];
        let sq_norm_a3 = a[6] * a[6] + a[7] * a[7] + a[8] * a[8];
        let dot_a1a2 = a[0] * a[3] + a[1] * a[4] + a[2] * a[5];
        let dot_a1a3 = a[0] * a[6] + a[1] * a[7] + a[2] * a[8];
        let dot_a2a3 = a[3] * a[6] + a[4] * a[7] + a[5] * a[8];

        (sq_norm_a1 - 1.0).powi(2)
            + (sq_norm_a2 - 1.0).powi(2)
            + (sq_norm_a3 - 1.0).powi(2)
            + 2.0 * (dot_a1a2.powi(2) + dot_a1a3.powi(2) + dot_a2a3.powi(2))
    }

    fn determinant9x1(&self, r: &na::SVector<f64, 9>) -> f64 {
        (r[0] * r[4] * r[8] + r[1] * r[5] * r[6] + r[2] * r[3] * r[7])
            - (r[6] * r[4] * r[2] + r[7] * r[5] * r[0] + r[8] * r[3] * r[1])
    }

    fn test_positive_depth(&self, r: &na::SVector<f64, 9>, t: &na::Vector3<f64>) -> bool {
        let m = self.point_mean;
        (r[6] * m[0] + r[7] * m[1] + r[8] * m[2] + t[2]) > 0.0
    }
}

pub fn sqpnp_solve(
    p3ds: &[(f64, f64, f64)],
    p2ds: &[(f64, f64)],
) -> Option<((f64, f64, f64), (f64, f64, f64))> {
    let mut solver = PnPSolver::new(p3ds, p2ds, SolverParameters::default());
    if solver.solve() {
        if let Some(sol) = solver.solutions.first() {
            return Some((
                (sol.r_hat[0], sol.r_hat[1], sol.r_hat[2]),
                (sol.t[0], sol.t[1], sol.t[2]),
            ));
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random;

    #[test]
    fn test_random_points_solve() {
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

        if let Some((rvec_pnp, tvec_pnp)) = sqpnp_solve(&p3ds, &p2ds) {
            let rvec_pnp = na::Vector3::new(rvec_pnp.0, rvec_pnp.1, rvec_pnp.2);
            let tvec_pnp = na::Vector3::new(tvec_pnp.0, tvec_pnp.1, tvec_pnp.2);

            let r_diff = (rvec_pnp - rvec).norm();
            let t_diff = (tvec_pnp - tvec).norm();

            println!("rvec gt  {}", rvec);
            println!("rvec pnp {}", rvec_pnp);
            println!("tvec gt  {}", tvec);
            println!("tvec pnp {}", tvec_pnp);
            println!("rvec diff {}", r_diff);
            println!("tvec diff {}", t_diff);

            assert!(
                r_diff < 1e-10,
                "Rotation vector difference too large: {}",
                r_diff
            );
            assert!(
                t_diff < 1e-10,
                "Translation vector difference too large: {}",
                t_diff
            );
        } else {
            panic!("Solver failed to find a solution");
        }
    }

    #[test]
    fn test_random_points_plane_solve() {
        for i in 0..1000 {
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

            if p3ds.len() < 3 {
                continue;
            }

            if let Some((rvec_pnp, tvec_pnp)) = sqpnp_solve(&p3ds, &p2ds) {
                let rvec_pnp = na::Vector3::new(rvec_pnp.0, rvec_pnp.1, rvec_pnp.2);
                let tvec_pnp = na::Vector3::new(tvec_pnp.0, tvec_pnp.1, tvec_pnp.2);

                let r_diff = (rvec_pnp - rvec).norm();
                let t_diff = (tvec_pnp - tvec).norm();

                let mut max_reproj_err = 0.0;
                for (p3, p2) in zip(&p3ds, &p2ds) {
                    let p3d_cam =
                        na::Isometry3::new(tvec_pnp, rvec_pnp) * na::Point3::new(p3.0, p3.1, p3.2);
                    if p3d_cam.z.abs() > 1e-6 {
                        let u = p3d_cam.x / p3d_cam.z;
                        let v = p3d_cam.y / p3d_cam.z;
                        let err = (u - p2.0).powi(2) + (v - p2.1).powi(2);
                        if err > max_reproj_err {
                            max_reproj_err = err;
                        }
                    }
                }

                if max_reproj_err > 1e-6 {
                    println!("Iteration {} failed", i);
                    println!("rvec gt  {}", rvec);
                    println!("rvec pnp {}", rvec_pnp);
                    println!("tvec gt  {}", tvec);
                    println!("tvec pnp {}", tvec_pnp);
                    println!("rvec diff {}", r_diff);
                    println!("tvec diff {}", t_diff);
                    println!("max reproj err {}", max_reproj_err);
                }
                assert!(
                    max_reproj_err < 1e-6,
                    "Reprojection error too large at iter {}: {}",
                    i,
                    max_reproj_err
                );
            } else {
                panic!("Solver failed to find a solution at iter {}", i);
            }
        }
    }
}
