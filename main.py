import cv2
import numpy as np
from typing import Tuple
from time import perf_counter
from scipy.spatial.transform import Rotation
from lsq_solver import LeastSquaresProblem

def sqpnp(p3ds: np.ndarray, p2ds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # tvec = tvec.reshape(3, 1)
    # rmat = Rotation.from_rotvec(rvec).as_matrix()
    # r_flat = rmat.flatten().reshape(-1, 1)

    Qs = []
    As = []
    Qsum = np.zeros((3, 3))
    QAsum = np.zeros((3, 9))
    for (p3d, p2d) in zip(p3ds, p2ds):
        p2d = np.array([p2d[0], p2d[1], 1.0]).reshape(3, 1)
        # t_p3d = rmat@p3d.reshape(3, 1) + tvec
        one_z = np.array([0, 0, 1]).reshape(1, 3)
        # s = one_z @ t_p3d
        # print(t_p3d.flatten(), (s * p2d).flatten(), t_p3d.flatten() - (s * p2d).flatten())
        # print(s*p2d, p3d)
        A = np.zeros((3, 9))
        A[0, 0:3] = p3d
        A[1, 3:6] = p3d
        A[2, 6:9] = p3d
        q = p2d @ one_z - np.eye(3, 3)
        print(q)
        Q = q.T @ q
        # art = A@r_flat+tvec
        # print(art.T @ Q @ art)
        As.append(A)
        Qs.append(Q)
        Qsum += Q
        QAsum += (Q@A)

    P = -1 * np.linalg.inv(Qsum) @ QAsum
    omega = np.zeros((9, 9))
    for (A, Q) in zip(As, Qs):
        omega += (A + P).T @ Q @ (A + P)
    # # print(omega)
    # print("r omega r")
    # print(r_flat.T @ omega @ r_flat)
    # rmat = Rotation.from_rotvec(np.zeros(3)).as_matrix()
    # r_flat = rmat.flatten().reshape(-1, 1)
    # # print(omega)
    # print(r_flat.T @ omega @ r_flat)
    problem = LeastSquaresProblem()
    def f(rr):
        rmat = Rotation.from_rotvec(rr).as_matrix()
        r_flat = rmat.flatten().reshape(-1, 1)
        # print(omega)
        return (r_flat.T @ omega @ r_flat).flatten()
    rv_init = np.zeros(3)
    problem.add_residual_block(1, f, rv_init)
    res = problem.solve(verbose=0)
    # print(res.x)
    rvec_solution = res.x
    tvec = P @ Rotation.from_rotvec(rvec_solution).as_matrix().reshape(9, 1)
    return (rvec_solution, tvec.flatten())

    
def main():
    rvec = np.random.random(3)
    tvec = np.random.random(3).reshape(3, 1)
    # transform = SE3(rvec, tvec)
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    n_points = 300
    p3ds = np.random.random((n_points, 3))
    p3ds[:, 2] += 0.5
    transformed_p3ds = (rmat @ p3ds.T + tvec).T
    noise = (np.random.random((n_points, 3)) - 0.5) / 100.0
    transformed_p3ds += noise
    p2ds = transformed_p3ds[:, :2] / transformed_p3ds[:, 2:3]

    t0 = perf_counter()
    _, pnp_rvec, pnp_tvec = cv2.solvePnP(p3ds, p2ds, np.eye(3, 3), np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    t1 = perf_counter()
    mine_rvec, mine_tvec = sqpnp(p3ds, p2ds)
    t2 = perf_counter()
    print(rvec, tvec.flatten())
    print(pnp_rvec.flatten(), pnp_tvec.flatten())
    print(mine_rvec.flatten(), mine_tvec.flatten())
    print(f"opencv: {t1 - t0}")
    print(f"mine: {t2 - t1}")
    pass

if __name__ == "__main__":
    main()