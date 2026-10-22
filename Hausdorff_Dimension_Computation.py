from __future__ import annotations

import math
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs


def spectral_radius(matrix, tolerance=1e-11):
    """Estimate the Perron eigenvalue of a nonnegative sparse matrix."""
    n = matrix.shape[0]
    if n == 1:
        return float(matrix[0, 0])
    # For tiny matrices, dense eigensolvers are more reliable than ARPACK.
    if n <= 4:
        return float(max(abs(np.linalg.eigvals(matrix.toarray()))))
    value = eigs(matrix, k=1, which="LM", v0=np.ones(n),
                 tol=tolerance, maxiter=max(3000, 20 * n),
                 return_eigenvectors=False)[0]
    return float(abs(value))
    """Estimate the Perron eigenvalue by power iteration (NumPy only)."""
    rows, cols, vals, n = matrix
    vector = np.ones(n, dtype=float)
    previous = None
    for _ in range(max(5000, 50 * n)):
        image = np.bincount(rows, weights=vals * vector[cols], minlength=n)
        scale = float(np.max(image))
        if scale == 0.0:
            return 0.0
        estimate = float(np.sum(image) / np.sum(vector))
        vector = image / scale
        # The ratio of successive 1-norms converges to the Perron eigenvalue.
        if previous is not None and abs(estimate - previous) <= tolerance * max(1.0, estimate):
            return estimate
        previous = estimate
    return estimate


def make_bounds_matrices(digits, n_intervals, s):
    """Construct A_s and B_s from equations (3.3)-(3.4) of the paper."""
    h = 1.0 / n_intervals
    x = np.linspace(0.0, 1.0, n_intervals + 1)
    gamma = min(digits)
    Gamma = max(digits)
    domain_right = 1.0 / gamma
    h = domain_right / n_intervals
    x = np.linspace(0.0, domain_right, n_intervals + 1)
    rows_a, cols_a, vals_a = [], [], []
    rows_b, cols_b, vals_b = [], [], []
    gamma = max(digits)

    for k, xk in enumerate(x):
        for m in digits:
            y = 1.0 / (xk + m)
            r = min(int(y / h), n_intervals - 1)
            left, right = x[r], x[r + 1]
            t = min(1.0, max(0.0, (y - left) / h))
            weight_left, weight_right = 1.0 - t, t

            
            product = max(0.0, (right - y) * (y - left))
            err1 = product * s * (2.0 * s + 1.0) * math.exp(2.0 * s * h)
            err2 = (product * s * (2.0 * s + 1.0) * (1.0 + gamma)
                    * math.exp(-2.0 * s * h))
            err1 = (product * s * (2.0 * s + 1.0) / (y * y)
                    * math.exp(2.0 * s * h / gamma))
            err2 = (product * s * (2.0 * s + 1.0)
                    / (2.0 / y + Gamma) ** 2
                    * math.exp(-2.0 * s * h / gamma))
            g_s = (xk + m) ** (-2.0 * s)
            factor_a = g_s * (1.0 - err1)
            factor_b = g_s * (1.0 - err2)
            if factor_a < -1e-13 or factor_b < -1e-13:
                raise ValueError(
                    "Mesh is too coarse: an approximation matrix has a negative entry. "
                    "Increase the number of intervals."
                )
            factor_a = max(0.0, factor_a)
            factor_b = max(0.0, factor_b)
            for col, weight in ((r, weight_left), (r + 1, weight_right)):
                if weight == 0.0:
                    continue
                rows_a.append(k); cols_a.append(col); vals_a.append(factor_a * weight)
                rows_b.append(k); cols_b.append(col); vals_b.append(factor_b * weight)

    shape = (n_intervals + 1, n_intervals + 1)
    A = coo_matrix((vals_a, (rows_a, cols_a)), shape=shape).tocsr()
    B = coo_matrix((vals_b, (rows_b, cols_b)), shape=shape).tocsr()
    size = n_intervals + 1
    A = (np.asarray(rows_a, dtype=int), np.asarray(cols_a, dtype=int),
         np.asarray(vals_a, dtype=float), size)
    B = (np.asarray(rows_b, dtype=int), np.asarray(cols_b, dtype=int),
         np.asarray(vals_b, dtype=float), size)
    return A, B


def bounds_at_s(digits, n_intervals, s):
    A, B = make_bounds_matrices(digits, n_intervals, s)
    return spectral_radius(A), spectral_radius(B)


def bisect_threshold(digits, n_intervals, which, tolerance=2e-9):
def bisect_threshold(digits, n_intervals, which, tolerance=2e-11):
    """Solve rho(A_s)=1 (lower dim) or rho(B_s)=1 (upper dim)."""
    lo, hi = 0.0, 1.0
    # rho(A_0)=rho(B_0)=|F|. Expand upper bracket if needed.
    def rho(s):
        a, b = bounds_at_s(digits, n_intervals, s)
        return a if which == "A" else b

    while rho(hi) > 1.0:
        hi *= 2.0
        if hi > 64:
            raise RuntimeError("Could not bracket the spectral-radius root.")
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if rho(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def read_digits():
    while True:
        try:
            count = int(input("Number of elements in F: ").strip())
            if count < 1:
                raise ValueError
            break
        except ValueError:
            print("Enter a positive integer.")

    digits = []
    for i in range(count):
        while True:
            try:
                m = int(input(f"Digit {i + 1} (positive integer): ").strip())
                if m < 1 or (digits and m <= digits[-1]):
                    raise ValueError
                digits.append(m)
                break
            except ValueError:
                print("Digits must be positive integers, entered in strictly increasing order.")
    return digits


def main():
    digits = read_digits()
    if len(digits) == 1:
        print(f"\nF = {digits}")
        print("Hausdorff dimension lower bound: 0.000000000000")
        print("Hausdorff dimension upper bound: 0.000000000000")
        print("A single continued-fraction map has a one-point invariant set.")
        return
    target_width = 1e-8
    n = 256
    while True:
        try:
            n = int(input("Number of mesh intervals (e.g. 200): ").strip())
            if n < 4:
                raise ValueError
            print(f"Calculating with {n} mesh intervals...", flush=True)
            lower = bisect_threshold(digits, n, "A")
            upper = bisect_threshold(digits, n, "B")
        except Exception as exc:
            print(f"Calculation failed: {exc}", file=sys.stderr)
            raise
        if upper - lower <= target_width:
            break
        except ValueError:
            print("Enter an integer of at least 4. More intervals usually give tighter bounds.")
        n *= 2
        if n > 32768:
            print("Could not reach the requested bound width with 32768 intervals. "
                  "The displayed interval is the best obtained.")
            break

    try:
        lower = bisect_threshold(digits, n, "A")
        upper = bisect_threshold(digits, n, "B")
    except Exception as exc:
        print(f"Calculation failed: {exc}", file=sys.stderr)
        raise

    print(f"\nF = {digits}")
    print(f"Mesh: {n} intervals (h = {1/n:.6g})")
    print(f"Hausdorff dimension lower bound: {lower:.12f}")
    print(f"Hausdorff dimension upper bound: {upper:.12f}")
    print(f"Mesh: {n} intervals (h = {1 / min(digits) / n:.6g})")
    print(f"Hausdorff dimension lower bound: {lower:.8f}")
    print(f"Hausdorff dimension upper bound: {upper:.8f}")
    print(f"Bound width:                   {upper - lower:.3e}")
    print("\nThe interval is the paper's interpolation bound; spectral radii are "
          "computed numerically in floating-point arithmetic.")


if __name__ == "__main__":
    main()
