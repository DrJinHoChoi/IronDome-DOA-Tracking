"""3D visualization for Iron Dome simulation."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_iron_dome_3d(threats, radar_sites, interceptor=None,
                      track_histories=None, title=None, save_path=None):
    """Full Iron Dome 3D visualization.

    Shows missile trajectories, radar positions, protected areas,
    and tracking results.

    Args:
        threats: List of Threat objects with trajectories.
        radar_sites: List of RadarSite objects.
        interceptor: Interceptor object (for protected areas).
        track_histories: Dict {track_id: state_history array} (optional).
        title: Plot title.
        save_path: File path to save figure.
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot missile trajectories
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(threats)))
    for i, threat in enumerate(threats):
        traj = threat.trajectory
        if len(traj) > 0:
            ax.plot(traj[:, 0] / 1000, traj[:, 1] / 1000, traj[:, 2] / 1000,
                    color=colors[i], alpha=0.5, linewidth=0.8)
            # Launch point
            ax.scatter(traj[0, 0] / 1000, traj[0, 1] / 1000, traj[0, 2] / 1000,
                       c='red', marker='^', s=15, alpha=0.6)

    # Plot radar positions
    for site in radar_sites:
        pos = site.position
        ax.scatter(pos[0] / 1000, pos[1] / 1000, pos[2] / 1000,
                   c='blue', marker='s', s=100, zorder=5,
                   label=site.name if site == radar_sites[0] else None)
        ax.text(pos[0] / 1000, pos[1] / 1000, pos[2] / 1000 + 0.5,
                site.name, fontsize=8, color='blue')

    # Plot protected areas
    if interceptor is not None:
        for center, radius in interceptor.protected_areas:
            theta = np.linspace(0, 2 * np.pi, 50)
            x_circle = center[0] / 1000 + radius / 1000 * np.cos(theta)
            y_circle = center[1] / 1000 + radius / 1000 * np.sin(theta)
            z_circle = np.zeros_like(theta)
            ax.plot(x_circle, y_circle, z_circle, 'g--', alpha=0.5, linewidth=1.5)

    # Plot track histories
    if track_histories is not None:
        track_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(track_histories)))
        for idx, (track_id, history) in enumerate(track_histories.items()):
            if len(history) > 1 and history.shape[1] >= 3:
                ax.plot(history[:, 0] / 1000, history[:, 1] / 1000,
                        history[:, 2] / 1000,
                        color=track_colors[idx % len(track_colors)],
                        linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title or 'Iron Dome Simulation - 3D View')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', alpha=0.6, label='Missile Trajectories'),
        Line2D([0], [0], color='blue', marker='s', linestyle='None',
               markersize=8, label='Radar Sites'),
        Line2D([0], [0], color='green', linestyle='--', label='Protected Areas'),
    ]
    if track_histories:
        legend_elements.append(
            Line2D([0], [0], color='blue', linestyle='--',
                   label='Estimated Tracks')
        )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_trajectories(threats, title=None, save_path=None):
    """Plot missile trajectories (top-down and side view).

    Args:
        threats: List of Threat objects.
        title: Plot title.
        save_path: File path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(threats)))

    # Top-down view (X-Y)
    ax = axes[0]
    for i, threat in enumerate(threats):
        traj = threat.trajectory
        if len(traj) > 0:
            ax.plot(traj[:, 0] / 1000, traj[:, 1] / 1000,
                    color=colors[i], alpha=0.5, linewidth=0.8)
            ax.scatter(traj[0, 0] / 1000, traj[0, 1] / 1000,
                       c='red', marker='^', s=10, alpha=0.5)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Top-Down View')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Side view (X-Z)
    ax = axes[1]
    for i, threat in enumerate(threats):
        traj = threat.trajectory
        if len(traj) > 0:
            ax.plot(traj[:, 0] / 1000, traj[:, 2] / 1000,
                    color=colors[i], alpha=0.5, linewidth=0.8)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Side View')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title or 'Missile Trajectories', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
