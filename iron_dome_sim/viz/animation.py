"""Real-time animation for Iron Dome simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class IronDomeAnimation:
    """Animated visualization of Iron Dome simulation.

    Shows real-time missile trajectories, radar tracking,
    and interception events.
    """

    def __init__(self, threats, radar_sites, threat_gen, dt=0.1,
                 interceptor=None):
        self.threats = threats
        self.radar_sites = radar_sites
        self.threat_gen = threat_gen
        self.dt = dt
        self.interceptor = interceptor

    def run(self, duration=60, interval=100, save_path=None):
        """Run the animation.

        Args:
            duration: Simulation duration in seconds.
            interval: Frame interval in milliseconds.
            save_path: File path to save animation (mp4/gif).
        """
        n_frames = int(duration / self.dt)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Top-down view
        ax_top = axes[0]
        ax_top.set_xlabel('X (km)')
        ax_top.set_ylabel('Y (km)')
        ax_top.set_title('Top-Down View')
        ax_top.grid(True, alpha=0.3)

        # Right: Side view
        ax_side = axes[1]
        ax_side.set_xlabel('X (km)')
        ax_side.set_ylabel('Altitude (km)')
        ax_side.set_title('Side View')
        ax_side.grid(True, alpha=0.3)

        # Static elements
        for site in self.radar_sites:
            pos = site.position / 1000
            ax_top.plot(pos[0], pos[1], 'bs', markersize=8)
            ax_side.plot(pos[0], pos[2], 'bs', markersize=8)

        if self.interceptor:
            for center, radius in self.interceptor.protected_areas:
                theta = np.linspace(0, 2 * np.pi, 50)
                ax_top.plot(
                    center[0]/1000 + radius/1000 * np.cos(theta),
                    center[1]/1000 + radius/1000 * np.sin(theta),
                    'g--', alpha=0.4
                )

        # Dynamic elements
        scat_top = ax_top.scatter([], [], c='red', s=15, alpha=0.7)
        scat_side = ax_side.scatter([], [], c='red', s=15, alpha=0.7)
        time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
        count_text = fig.text(0.5, 0.96, '', ha='center', fontsize=11)

        # Pre-draw faint full trajectories
        for threat in self.threats:
            traj = threat.trajectory
            if len(traj) > 0:
                ax_top.plot(traj[:, 0]/1000, traj[:, 1]/1000,
                            'r-', alpha=0.05, linewidth=0.5)
                ax_side.plot(traj[:, 0]/1000, traj[:, 2]/1000,
                             'r-', alpha=0.05, linewidth=0.5)

        def init():
            scat_top.set_offsets(np.empty((0, 2)))
            scat_side.set_offsets(np.empty((0, 2)))
            return scat_top, scat_side, time_text, count_text

        def update(frame):
            t = frame * self.dt
            positions, active_ids = self.threat_gen.get_positions_at_time(
                self.threats, t
            )

            if len(positions) > 0:
                scat_top.set_offsets(positions[:, :2] / 1000)
                side_data = np.column_stack([positions[:, 0], positions[:, 2]]) / 1000
                scat_side.set_offsets(side_data)
            else:
                scat_top.set_offsets(np.empty((0, 2)))
                scat_side.set_offsets(np.empty((0, 2)))

            time_text.set_text(f't = {t:.1f} s')
            count_text.set_text(f'Active threats: {len(active_ids)}')

            return scat_top, scat_side, time_text, count_text

        anim = FuncAnimation(fig, update, frames=n_frames,
                             init_func=init, blit=True,
                             interval=interval)

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=10)
            else:
                anim.save(save_path, writer='ffmpeg', fps=30)
        else:
            plt.show()

        return anim
