import numpy as np
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
from QuCOOP.src import psr, utils

if __name__ == "__main__":
    ref_color, tmp_color = 'b', 'y'

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(3, 1)
    fig.tight_layout(pad=-3)
    ref_points = np.loadtxt('../data/Fish/fish_teaser.dat', delimiter=',').T
    tmp_points = ref_points.copy()
    size = 7

    ref_points -= ref_points.mean(axis=1, keepdims=True)
    tmp_points -= tmp_points.mean(axis=1, keepdims=True)

    a, b = ref_points.shape[1], tmp_points.shape[1]
    n = np.maximum(a, b)
    ref_points = np.pad(ref_points, ((0, 0), (0, n - a)), 'constant', constant_values=((0, 0), (0, 0)))
    tmp_points = np.pad(tmp_points, ((0, 0), (0, n - b)), 'constant', constant_values=((0, 0), (0, 0)))

    theta = np.deg2rad(np.random.randint(-80, 80))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    P = np.eye(n)
    np.random.shuffle(P)
    tmp_points = R @ tmp_points
    ref_points = ref_points @ P

    colors = list(map(lambda i: "#" + "%06x" % np.random.randint(0, 0xFFFFFF), range(n)))

    # Set seed back to None
    np.random.seed(None)

    # Sols star
    R_star = R.T
    P_star = P.T

    # Initial
    R_rec_init = np.eye(2)
    P_rec_init = np.eye(n)

    # Penalty factors
    alpha, beta = .1, 1   # Adjust alpha

    # Our solver
    psr = psr.PointSetRegistration(ref_points=ref_points, tmp_points=tmp_points, alpha=alpha, beta=beta, R_star=R_star, P_star=P_star)
    history = psr.solve(K=5, maxiter=15, simulated_anneal=True)

    R_rec_ours, P_rec_ours = psr.R_sol, psr.P_sol

    # CPD
    reg = RigidRegistration(X=ref_points[:, :a].T, Y=tmp_points[:, :b].T)
    TY, (s_reg, R_reg, t_reg) = reg.register()
    R_rec_cpd = R_reg.T
    P_rec_cpd = reg.P.T

    # Result

    # Plots
    # Initial
    ref_points_plot = ref_points @ P_rec_init
    tmp_points_plot = R_rec_init @ tmp_points
    ref_points_plot = ref_points_plot[:, :a]
    tmp_points_plot = tmp_points_plot[:, :b]
    for i, c in zip(list(range(n)), colors):
        try:
            ax[0].scatter(*tmp_points_plot[:, i], color=tmp_color, marker='s', s=size)
            ax[0].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
            ax[0].plot([ref_points_plot[0, i], tmp_points_plot[0, i]], [ref_points_plot[1, i], tmp_points_plot[1, i]], c='gray', linewidth=.5, alpha=.5)
        except:
            ax[0].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
    ax[0].axis('equal')
    ax[0].axis(False)
    x_lim, y_lim = ax[0].get_xlim(), ax[0].get_ylim()
    x_lim = [x + y for (x, y) in zip([-.5, +5], x_lim)]
    y_lim = [x + y for (x, y) in zip([-.5, +5], y_lim)]

    # Ours
    ref_points_plot = ref_points @ P_rec_ours
    tmp_points_plot = R_rec_ours @ tmp_points
    ref_points_plot = ref_points_plot[:, :a]
    tmp_points_plot = tmp_points_plot[:, :b]
    for i, c in zip(list(range(n)), colors):
        try:
            ax[1].scatter(*tmp_points_plot[:, i], color=tmp_color, marker='s', s=size)
            ax[1].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
            ax[1].plot([ref_points_plot[0, i], tmp_points_plot[0, i]], [ref_points_plot[1, i], tmp_points_plot[1, i]], c='gray', linewidth=.5, alpha=.5)
        except:
            ax[1].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
    ax[1].set_title(None)
    ax[1].axis('equal')
    ax[1].axis(False)

    # CPD
    ref_points_plot = ref_points @ P_rec_cpd
    tmp_points_plot = s_reg * R_rec_cpd @ tmp_points + t_reg[:, None]
    ref_points_plot = ref_points_plot[:, :a]
    tmp_points_plot = tmp_points_plot[:, :b]
    for i, c in zip(list(range(n)), colors):
        try:
            ax[2].scatter(*tmp_points_plot[:, i], color=tmp_color, marker='s', s=size)
            ax[2].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
            ax[2].plot([ref_points_plot[0, i], tmp_points_plot[0, i]], [ref_points_plot[1, i], tmp_points_plot[1, i]], c='gray', linewidth=.5, alpha=.5)
        except:
            ax[2].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
    ax[2].set_title(None)
    ax[2].axis('equal')
    ax[2].axis(False)
    plt.show()
