import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import laplace
from tqdm import tqdm

def grad(field):
    """
    Computes the gradient of a 2D scalar field.

    Args:
        field (numpy.ndarray): Input scalar field of shape [H, W].

    Returns:
        tuple: Gradients along x-axis and y-axis (grad_x, grad_y).
    """
    grad_x = np.gradient(field, axis=0)  # Gradient along x-axis
    grad_y = np.gradient(field, axis=1)  # Gradient along y-axis
    return grad_x, grad_y

def div(grad_x, grad_y):
    """
    Computes the divergence of a 2D vector field.

    Args:
        grad_x (numpy.ndarray): Gradient field along x-axis of shape [H, W].
        grad_y (numpy.ndarray): Gradient field along y-axis of shape [H, W].

    Returns:
        numpy.ndarray: Divergence of the field of shape [H, W].
    """
    div_x = np.gradient(grad_x, axis=0)  # Divergence along x-axis
    div_y = np.gradient(grad_y, axis=1)  # Divergence along y-axis
    return div_x + div_y

# 0 : normal cahn hilliard
def cahn_hilliard_simulation(scalar_field, D=1.0, gamma=1.0, dt=0.01, steps=100, tqdm_log=False, make_gif=False, gif_path="cahn_hilliard.gif"):
    """
    Simulates the Cahn-Hilliard equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        laplacian_c = div(grad_cx, grad_cy)
        mu = - gamma * laplacian_c + (c[0, 0]**3 - c[0, 0])
        grad_mux, grad_muy = grad(mu)
        laplacian_mu = div(grad_mux, grad_muy)
        dc_dt = D * laplacian_mu
        c[0, 0] += dt * dc_dt

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 1 : fixed cahn hilliard
def fixed_cahn_hilliard_simulation(scalar_field, D=1.0, gamma=1.0, dt=0.01, steps=100, threshold=0.90, tqdm_log=False, make_gif=False, gif_path="cahn_hilliard.gif"):
    """
    Simulates the Cahn-Hilliard equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        old_c = c[0, 0].copy()
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        laplacian_c = div(grad_cx, grad_cy)
        mu = - gamma * laplacian_c + (c[0, 0]**3 - c[0, 0])
        grad_mux, grad_muy = grad(mu)
        laplacian_mu = div(grad_mux, grad_muy)
        dc_dt = D * laplacian_mu
        c[0, 0] += dt * dc_dt
        # Fix the values outside the threshold
        mask = (-threshold > old_c) | (old_c > threshold)
        c[0, 0][mask] = old_c[mask]

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 2 : positive_fixed cahn hilliard
def positive_fixed_cahn_hilliard_simulation(scalar_field, D=1.0, gamma=1.0, dt=0.01, steps=100, threshold=0.90, tqdm_log=False, make_gif=False, gif_path="cahn_hilliard.gif"):
    """
    Simulates the Cahn-Hilliard equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        old_c = c[0, 0].copy()
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        laplacian_c = div(grad_cx, grad_cy)
        mu = - gamma * laplacian_c + (c[0, 0]**3 - c[0, 0])
        grad_mux, grad_muy = grad(mu)
        laplacian_mu = div(grad_mux, grad_muy)
        dc_dt = D * laplacian_mu
        c[0, 0] += dt * dc_dt
        # Fix the values outside the threshold
        mask = (old_c > threshold)
        c[0, 0][mask] = old_c[mask]

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 3 : fixed positive decrease
def fixed_positive_decrease(scalar_field, D=1.0, gamma=1.0, dt=0.01, steps=100, threshold=0.90, tqdm_log=False, make_gif=False, gif_path="cahn_hilliard.gif"):
    """
    Simulates the Cahn-Hilliard equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        old_c = c[0, 0].copy()
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        laplacian_c = div(grad_cx, grad_cy)
        mu = - gamma * laplacian_c + (c[0, 0]**3 - c[0, 0])
        dc_dt = D * (-mu)
        c[0, 0] += dt * dc_dt
        # Fix the values outside the threshold
        mask = (old_c > threshold)
        c[0, 0][mask] = old_c[mask]

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 4 : Allen-Cahn
def allen_cahn_simulation(scalar_field, D=1.0, gamma=1.0, dt=0.01, steps=100, tqdm_log=False, make_gif=False, gif_path="allen_cahn.gif"):
    """
    Simulates the Allen-Cahn equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        dc_dt = D * (div(gamma * grad_cx, gamma * grad_cy) - (c[0, 0]**3 - c[0, 0]))
        
        c[0, 0] += dt * dc_dt

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 5 : positive fixed allen cahn
def positive_allen_cahn_simulation(scalar_field, D=1.0, gamma=1.0, dt=0.01, steps=100, threshold=0.90, tqdm_log=False, make_gif=False, gif_path="allen_cahn.gif"):
    """
    Simulates the Allen-Cahn equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        old_c = c[0, 0].copy()
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        dc_dt = D * (div(gamma * grad_cx, gamma * grad_cy) - (c[0, 0]**3 - c[0, 0]))
        
        c[0, 0] += dt * dc_dt
        mask = (old_c > threshold)
        c[0, 0][mask] = old_c[mask]

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 6 : curvature drive Allen-Cahn
def curvature_drive_allen_cahn(scalar_field, D=1.0, gamma=1.0, alpha=1.0, dt=0.01, steps=100, tqdm_log=False, make_gif=False, gif_path="allen_cahn.gif"):
    """
    Simulates the Allen-Cahn equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        alpha (float): Parameter for the curvature term.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        norm = np.sqrt(grad_cx**2 + grad_cy**2 + 1e-10)
        kappa = div(grad_cx / norm, grad_cy / norm)
        dc_dt = D * (div(gamma * grad_cx, gamma * grad_cy) - (c[0, 0]**3 - c[0, 0])) + alpha * kappa
        
        c[0, 0] += dt * dc_dt

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c

# 7 : curvature drive cahn hilliard
def curvature_drive_cahn_hilliard_simulation(scalar_field, D=1.0, gamma=1.0, alpha=1.0, dt=0.01, steps=100, tqdm_log=False, make_gif=False, gif_path="cahn_hilliard.gif"):
    """
    Simulates the Cahn-Hilliard equation.

    Parameters:
        scalar_field (numpy.ndarray): Initial scalar field of shape [1, 1, H, W] with values in [-1, 1].
        D (float): Diffusion coefficient.
        gamma (float): Parameter for the chemical potential term.
        dt (float): Time step for simulation.
        steps (int): Number of simulation steps.
        make_gif (bool): Whether to generate a GIF of the simulation.
        gif_path (str): File path to save the GIF if make_gif is True.

    Returns:
        numpy.ndarray: The evolved scalar field after the simulation.
    """
    assert scalar_field.ndim == 4 and scalar_field.shape[0] == 1 and scalar_field.shape[1] == 1, \
        "Input must be of shape [1, 1, H, W]"
    
    H, W = scalar_field.shape[2], scalar_field.shape[3]
    c = scalar_field.copy()
    frames = [] if make_gif else None

    # Step size for recording frames
    record_interval = max(steps // 100, 1)

    tbar = tqdm(range(steps), ncols=80) if tqdm_log else range(steps)

    for step in tbar:
        grad_cx, grad_cy = grad(c[0, 0])  # Only the spatial dimensions
        laplacian_c = div(grad_cx, grad_cy)
        mu = - gamma * laplacian_c + (c[0, 0]**3 - c[0, 0])
        norm = np.sqrt(grad_cx**2 + grad_cy**2 + 1e-10)
        kappa = div(grad_cx / norm, grad_cy / norm)
        
        grad_mux, grad_muy = grad(mu - alpha * kappa)
        laplacian_mu = div(grad_mux, grad_muy)
        dc_dt = D * laplacian_mu
        c[0, 0] += dt * dc_dt

        # Record frames for GIF if required
        if make_gif and (step % record_interval == 0 or step == steps - 1):
            frames.append((step, c[0, 0].copy()))

    # Generate GIF if requested
    if make_gif:
        _generate_gif(frames, gif_path)
    
    return c


# select pde
def select_pde(mode):
    if mode == 0:
        return cahn_hilliard_simulation
    elif mode == 1:
        return fixed_cahn_hilliard_simulation
    elif mode == 2:
        return positive_fixed_cahn_hilliard_simulation
    elif mode == 3:
        return fixed_positive_decrease
    elif mode == 4:
        return allen_cahn_simulation
    elif mode == 5:
        return positive_allen_cahn_simulation
    elif mode == 6:
        return curvature_drive_allen_cahn
    elif mode == 7:
        return curvature_drive_cahn_hilliard_simulation





def _generate_gif(frames, gif_path):
    """
    Helper function to generate a GIF from simulation frames.

    Parameters:
        frames (list): List of (step, scalar_field) tuples.
        gif_path (str): Path to save the GIF.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0][1], cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    # Initialize a legend-like text
    legend_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    def update(frame):
        frame_idx, (step, field) = frame  # frame_idx is the index, (step, field) is the tuple
        im.set_array(field)
        legend_text.set_text(f"Step: {step}")  # Update with step and frame index
        return im, legend_text

    # Pass frames with their indices
    plt.tight_layout()
    ani = FuncAnimation(fig, update, frames=enumerate(frames), blit=True, interval=50, save_count=len(frames))
    ani.save(gif_path, writer=PillowWriter(fps=10), dpi=150)
    plt.close()

