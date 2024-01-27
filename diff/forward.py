# from https://colab.research.google.com/drive/
# 1aSQTgoqmyqGpLI9q7IRDlXXeMdAG-E4X?usp=sharing#scrollTo=1iwJfowMYB5f
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

# Simulate SDE with drift function f and
# noise amplitude g for arbitrary number of steps


def forward_SDE_simulation(x0, nsteps, dt, f, g, params):

    # Initialize time and a stochastic trajectory
    t = 0
    x_traj = np.zeros((nsteps + 1, *x0.shape))
    x_traj[0] = np.copy(x0)

    # Perform many Euler-Maruyama time steps
    for i in range(nsteps):
        # standard normal random number
        random_normal = np.random.randn(*x0.shape)

        x_traj[i+1] = x_traj[i] + f(x_traj[i], t, params)*dt + \
            g(x_traj[i], t, params)*np.sqrt(dt)*random_normal
        t = t + dt

    return x_traj


# Drift function for diffusion (returns zeros)
def f_diff_simple(x, t, params):
    return np.zeros((*x.shape,))


# Noise amplitude for diffusion (constant)
def g_diff_simple(x, t, params):
    sigma = params['sigma']
    return sigma*np.ones((*x.shape,))


# Exact transition probability for 1D diffusion
def transition_probability_diffusion_exact(x, t, params):
    x0, sigma = params['x0'], params['sigma']

    # pdf of normal distribution with mean x0 and variance (sigma^2)*t
    pdf = norm.pdf(x, loc=x0, scale=np.sqrt((sigma**2)*t))
    return pdf


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

    return x_traj, t, params, T


def plot(x_traj, t, params, T):
    # Plot initial distribution (distribution before diffusion)
    plt.hist(x_traj[0], density=True, bins=100)
    plt.title("$t = 0$", fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("probability", fontsize=20)
    plt.savefig("t0.png")

    # Compute exact transition probability
    x_f_min, x_f_max = np.amin(x_traj[-1]), np.amax(x_traj[-1])
    num_xf = 1000
    x_f_arg = np.linspace(x_f_min, x_f_max, num_xf)
    pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

    # Plot final distribution (distribution after diffusion)
    plt.hist(x_traj[-1], bins=100, density=True)
    plt.plot(x_f_arg, pdf_final, color='black', linewidth=5)
    plt.title("$t = $"+str(T), fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("probability", fontsize=20)
    plt.savefig("t2.png")

    # Plot some trajectories
    sample_trajectories = [0, 1, 2, 3, 4]
    for s in sample_trajectories:
        plt.plot(t, x_traj[:, s])
    plt.title("Sample trajectories", fontsize=20)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("x", fontsize=20)
    plt.savefig("trajectories.png")


if __name__ == "__main__":
    x_traj, t, params, T = simulate()
    plot(x_traj, t, params, T)
