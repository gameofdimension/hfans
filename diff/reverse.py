# Drift function for a reverse process. More general than 1D diffusion.
from matplotlib import pyplot as plt
import numpy as np

from forward import (
    f_diff_simple,
    forward_SDE_simulation,
    g_diff_simple,
    transition_probability_diffusion_exact
)


def f_diff_simple_rev(x, t, params):
    T = params['T']
    return - f_diff_simple(x, T-t, params) + \
        (g_diff_simple(x, T-t, params)**2)*score(x, T-t, params)


# Noise function for 1D diffusion (constant)
def g_diff_simple_rev(x, t, params):
    sigma = params['sigma']
    return sigma*np.ones(len(x))


# The score function for 1D diffusion from a point source.
# Score = grad log p(x,t)
def score(x, t, params):
    x0 = params['x0']
    sigma = params['sigma']

    score_ = (x0 - x)/((sigma**2)*(t))

    return score_


def simulate():
    sigma = 1         # noise amplitude for 1D diffusion

    num_samples = 1000
    x0 = np.zeros(num_samples)    # initial condition for diffusion

    nsteps = 2000      # number of simulation steps
    dt = 0.001          # size of small time steps
    T = nsteps*dt
    t = np.linspace(0, T, nsteps + 1)

    params = {'sigma': sigma, 'x0': x0, 'T': T}
    x_traj = forward_SDE_simulation(
        x0, nsteps, dt, f_diff_simple, g_diff_simple, params)
    x_traj_rev = forward_SDE_simulation(
        x_traj[-1], nsteps, dt, f_diff_simple_rev, g_diff_simple_rev, params)

    return x_traj, x_traj_rev, t, params, T


def plot(x_traj, x_traj_rev, t, params, T):
    # Compute exact transition probability
    x_f_min, x_f_max = np.amin(x_traj[-1]), np.amax(x_traj[-1])
    num_xf = 1000
    x_f_arg = np.linspace(x_f_min, x_f_max, num_xf)
    pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

    # Plot final distribution (distribution
    # after diffusion / before reverse diffusion)
    fig, ax = plt.subplots()
    ax.hist(x_traj[-1], bins=100, density=True)
    ax.plot(x_f_arg, pdf_final, color='black', linewidth=5)
    ax.set_title("$t = $"+str(T), fontsize=20)
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("probability", fontsize=20)
    plt.savefig('output/rev_t_final.png')

    # Plot initial distribution (distribution
    # before diffusion / after reverse diffusion)
    # fig, ax = plt.subplots(1, 2, width=)
    fig, ax = plt.subplots()
    ax.hist(x_traj_rev[-1], density=True, bins=100, label='rev. diff.')
    ax.hist(x_traj[0], density=True, bins=100, label='true')

    ax.set_title("$t = 0$", fontsize=20)
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("probability", fontsize=20)
    ax.legend(fontsize=15)
    plt.savefig('output/rev_t_init.png')

    # Plot some trajectories
    fig, ax = plt.subplots()
    sample_trajectories = [0, 1, 2, 3, 4]
    for s in sample_trajectories:
        ax.plot(t, x_traj_rev[:, s])
    ax.set_title("Sample trajectories (reverse process)", fontsize=20)
    ax.set_xlabel("$t$", fontsize=20)
    ax.set_ylabel("x", fontsize=20)
    plt.savefig('output/rev_trajectories.png')


if __name__ == '__main__':
    x_traj, x_traj_rev, t, params, T = simulate()
    plot(x_traj, x_traj_rev, t, params, T)
