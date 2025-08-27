
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Objective Function ---------------------------
# A 2D test function: rotated ellipsoidal bowl + mild sinusoidal ripple
def make_objective(seed=0):
    rng = np.random.default_rng(seed)
    # Random rotation matrix
    theta = rng.uniform(0.0, np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    # Ellipse scales
    s = np.diag([1.0, 10.0])  # ill-conditioning
    Q = R @ s @ R.T

    def f(x):
        x = np.asarray(x).reshape(2)
        quad = x.T @ Q @ x
        ripple = 0.1*np.sin(3.0*x[0]) * np.cos(3.0*x[1])
        return quad + ripple
    return f

# --------------------------- Random Directions ---------------------------
def random_directions_optimize(f, x0, step0=0.5, iters=100, P=8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float).reshape(2)
    s = float(step0)
    hist = [x.copy()]
    best_val = f(x)
    for t in range(iters):
        # sample P random unit directions
        U = rng.normal(size=(P, 2))
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12

        candidates = []
        for u in U:
            for sign in (+1.0, -1.0):
                x_cand = x + sign * s * u
                val = f(x_cand)
                candidates.append((val, x_cand))

        # choose best candidate
        candidates.sort(key=lambda p: p[0])
        best_val_new, x_new = candidates[0]

        if best_val_new < best_val:
            x = x_new
            best_val = best_val_new
            # optional short line search along the same direction (one extra step)
            d = (x_new - hist[-1])
            if np.linalg.norm(d) > 0:
                x_ls = x + d
                val_ls = f(x_ls)
                if val_ls < best_val:
                    x, best_val = x_ls, val_ls
        else:
            # shrink step if no improvement
            s *= 0.7

        hist.append(x.copy())
    return np.array(hist), best_val

# --------------------------- Minimal CMA-ES -------------------------------
# Based on Hansen's CMA-ES tutorial (simplified for 2D demo)
def cma_es_optimize(f, m0, sigma0=0.7, iters=80, lam=16, seed=1):
    rng = np.random.default_rng(seed)
    N = 2
    m = np.array(m0, dtype=float).reshape(N)
    sigma = float(sigma0)
    C = np.eye(N)
    ps = np.zeros(N)
    pc = np.zeros(N)

    # weights
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)

    # parameters
    c_sigma = (mu_eff + 2) / (N + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(N + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff/N) / (N + 4 + 2*mu_eff/N)
    c1 = 2 / ((N + 1.3)**2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)**2 + mu_eff))
    chiN = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))

    hist = [m.copy()]
    best_val = f(m)

    for t in range(iters):
        # sample offspring
        A = np.linalg.cholesky(C)
        zs = rng.normal(size=(lam, N))
        xs = m + sigma * (zs @ A.T)
        fs = np.array([f(x) for x in xs])

        # select best mu
        idx = np.argsort(fs)
        xs_mu = xs[idx[:mu]]
        zs_mu = zs[idx[:mu]]
        fs_mu = fs[idx[:mu]]

        # recombination
        m_new = np.sum(weights[:, None] * xs_mu, axis=0)
        z_w = np.sum(weights[:, None] * zs_mu, axis=0)

        # update evolution path for sigma
        C_inv_sqrt = np.linalg.inv(A).T  # since A is chol(C)
        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ z_w)

        # step-size control
        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1))

        # hsig indicator
        hsig = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(t+1))) < (1.4 + 2/(N+1)) * chiN) else 0.0

        # update evolution path for covariance
        y = m_new - m
        pc = (1 - c_c) * pc + hsig * np.sqrt(c_c*(2 - c_c)*mu_eff) * (y / (sigma + 1e-12))

        # rank-one and rank-mu updates
        C = (1 - c1 - c_mu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c_c*(2 - c_c) * C)
        for i in range(mu):
            zi = zs_mu[i]
            C += c_mu * weights[i] * np.outer(zi, zi)

        m = m_new
        hist.append(m.copy())
        best_val = min(best_val, f(m))

    return np.array(hist), best_val

# --------------------------- Plotting Utilities ---------------------------
def plot_contours_and_path(f, path, title, filename):
    # grid for contours
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, Z, levels=30)
    # Trajectory
    ax.plot(path[:,0], path[:,1], marker='o', linewidth=2, markersize=3)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    return fig, ax

# --------------------------- Main Demo ------------------------------------
def main(seed=0):
    f = make_objective(seed=seed)
    x0 = np.array([2.5, 2.0])

    # RD
    rd_path, rd_best = random_directions_optimize(f, x0, step0=0.8, iters=100, P=8, seed=seed)
    plot_contours_and_path(f, rd_path, "Random Directions trajectory", "rd_trajectory.png")

    # CMA-ES
    cma_path, cma_best = cma_es_optimize(f, x0, sigma0=0.8, iters=80, lam=16, seed=seed+1)
    plot_contours_and_path(f, cma_path, "CMA-ES mean trajectory", "cmaes_trajectory.png")

    print("RD final point:", rd_path[-1], "best value:", rd_best)
    print("CMA-ES final mean:", cma_path[-1], "best value:", cma_best)
    print("Saved figures: rd_trajectory.png, cmaes_trajectory.png")

if __name__ == "__main__":
    main()
