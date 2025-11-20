
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import imageio.v3 as imageio  # modern API
import os

os.makedirs('gif_frames', exist_ok=True)

# Parameters for geometric Brownian motion (GBM)
S0 = 1.0    # initial stock price
mu = 0.2    # drift rate (increased for noticeable drift)
sigma = 0.2 # volatility

num_frames = 100
t_values = np.linspace(0.1, 5, num_frames)  # avoid t=0 to prevent singularity in density
S_range = np.linspace(0.01, 5, 200)         # price range

# Simulate GBM sample paths
num_sample_paths = 5
sample_paths = np.zeros((num_sample_paths, len(t_values)))
dt = t_values[1] - t_values[0]
for i in range(num_sample_paths):
    increments = np.random.normal(0, np.sqrt(dt), size=len(t_values)-1)
    W = np.concatenate(([0], np.cumsum(increments)))
    sample_paths[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t_values + sigma * W)

frames = []

for i, t in enumerate(t_values):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    mask = t_values <= t
    T_sub, S_sub = np.meshgrid(t_values[mask], S_range)
    P_sub = (1 / (S_sub * sigma * np.sqrt(2 * np.pi * T_sub))) * \
            np.exp(- (np.log(S_sub / S0) - (mu - 0.5 * sigma**2) * T_sub)**2 / (2 * sigma**2 * T_sub))
    ax.plot_surface(S_sub, T_sub, P_sub, cmap='viridis', alpha=0.7, edgecolor='none')
    
    for sp in sample_paths:
        
        ax.plot(sp[:i+1], t_values[:i+1], np.zeros(i+1), 'r-', marker='o', markersize=3)
    
    ax.set_xlabel('Stock Price S')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Density')
    ax.set_title(f'Geometric Brownian Motion at t = {t:.2f}')
    ax.view_init(elev=30, azim=-60)
    
    frame_path = f'gif_frames/geometric_brownian_drifted_3d_t_{t:.2f}.png'
    plt.savefig(frame_path)
    plt.close()
    frames.append(imageio.imread(frame_path))

imageio.imwrite('geometric_brownian_drifted_3d.gif', frames, duration=0.1)
