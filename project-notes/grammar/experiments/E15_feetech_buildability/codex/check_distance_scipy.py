"""Optional independent check of the NumPy OBB distance implementation.

Run with python3 check_distance_scipy.py. No files are changed.
SciPy solves the convex closest-points problem over two local box coordinates.
"""
import warnings
warnings.filterwarnings('ignore', message='A NumPy version')
import numpy as np
from scipy.optimize import minimize
from feetech_geometry import H, unit, rotation, box_distance


def main():
    rng = np.random.default_rng(7)
    errors = []
    for _ in range(50):
        A = rotation(unit(rng.normal(size=3)), rng.uniform(-180, 180))
        B = rotation(unit(rng.normal(size=3)), rng.uniform(-180, 180))
        d = unit(rng.normal(size=3))*rng.uniform(10, 80)
        M = np.column_stack((A, -B))
        def objective(t):
            r = M@t-d
            return .5*(r@r), M.T@r
        result = minimize(objective, np.zeros(6), jac=True,
                          bounds=list(zip(np.tile(-H, 2), np.tile(H, 2))),
                          method='L-BFGS-B',
                          options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 1000})
        reference = np.sqrt(max(0, 2*result.fun))
        actual = box_distance(np.zeros(3), A, d, B)
        errors.append(abs(actual-reference))
    assert max(errors) < 1e-5, max(errors)
    print(f'PASS: 50 independent convex-QP checks; largest error {max(errors):.3g} mm')


if __name__ == '__main__':
    main()
