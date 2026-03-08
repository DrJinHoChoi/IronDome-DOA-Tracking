"""Motion models for target tracking.

Defines state transition and observation models for various
target dynamics: constant velocity, constant acceleration,
and ballistic trajectory.
"""

import numpy as np


class ConstantVelocity:
    """Constant Velocity (CV) motion model.

    State: x = [theta_az, theta_el, theta_dot_az, theta_dot_el]
    (DOA-only tracking in angular domain)
    """

    dim_state = 4
    dim_obs = 2  # [theta_az, theta_el]

    def __init__(self, dt=0.1, process_noise_std=0.01):
        """
        Args:
            dt: Time step in seconds.
            process_noise_std: Process noise standard deviation (rad/s^2).
        """
        self.dt = dt
        self.q = process_noise_std

    def F(self, x=None):
        """State transition matrix."""
        dt = self.dt
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

    def Q(self, x=None):
        """Process noise covariance."""
        dt = self.dt
        q = self.q ** 2
        return q * np.array([
            [dt**3/3, 0,       dt**2/2, 0      ],
            [0,       dt**3/3, 0,       dt**2/2],
            [dt**2/2, 0,       dt,      0      ],
            [0,       dt**2/2, 0,       dt     ],
        ])

    def H(self, x=None):
        """Observation matrix (linear: observe angles directly)."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

    def R(self, measurement_noise_std=0.02):
        """Measurement noise covariance.

        Args:
            measurement_noise_std: Noise std in radians (~1 degree).
        """
        r = measurement_noise_std ** 2
        return r * np.eye(2)

    def predict(self, x):
        """Predict next state."""
        return self.F() @ x

    def observe(self, x):
        """Observation function."""
        return self.H() @ x


class ConstantAcceleration:
    """Constant Acceleration (CA) motion model.

    State: x = [theta_az, theta_el, theta_dot_az, theta_dot_el,
                theta_ddot_az, theta_ddot_el]
    """

    dim_state = 6
    dim_obs = 2

    def __init__(self, dt=0.1, process_noise_std=0.005):
        self.dt = dt
        self.q = process_noise_std

    def F(self, x=None):
        dt = self.dt
        return np.array([
            [1, 0, dt, 0,  dt**2/2, 0      ],
            [0, 1, 0,  dt, 0,       dt**2/2],
            [0, 0, 1,  0,  dt,      0      ],
            [0, 0, 0,  1,  0,       dt     ],
            [0, 0, 0,  0,  1,       0      ],
            [0, 0, 0,  0,  0,       1      ],
        ])

    def Q(self, x=None):
        dt = self.dt
        q = self.q ** 2
        # Discrete white noise acceleration model
        G = np.array([
            [dt**2/2, 0      ],
            [0,       dt**2/2],
            [dt,      0      ],
            [0,       dt     ],
            [1,       0      ],
            [0,       1      ],
        ])
        return q * G @ G.T

    def H(self, x=None):
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])

    def R(self, measurement_noise_std=0.02):
        return measurement_noise_std ** 2 * np.eye(2)

    def predict(self, x):
        return self.F() @ x

    def observe(self, x):
        return self.H() @ x


class BallisticModel:
    """Ballistic trajectory model in 3D Cartesian coordinates.

    State: x = [px, py, pz, vx, vy, vz]
    Observation: z = [theta_az, theta_el] (nonlinear)

    Used with EKF/UKF for tracking missiles with gravity.
    """

    dim_state = 6
    dim_obs = 2
    GRAVITY = 9.81  # m/s^2

    def __init__(self, dt=0.1, process_noise_std=1.0, radar_pos=None):
        """
        Args:
            dt: Time step in seconds.
            process_noise_std: Process noise std in m/s^2.
            radar_pos: Radar position [x, y, z] in meters.
        """
        self.dt = dt
        self.q = process_noise_std
        self.radar_pos = np.array(radar_pos) if radar_pos is not None else np.zeros(3)

    def f(self, x):
        """Nonlinear state transition (ballistic with gravity)."""
        dt = self.dt
        px, py, pz, vx, vy, vz = x
        return np.array([
            px + vx * dt,
            py + vy * dt,
            pz + vz * dt - 0.5 * self.GRAVITY * dt ** 2,
            vx,
            vy,
            vz - self.GRAVITY * dt,
        ])

    def F(self, x=None):
        """Jacobian of state transition."""
        dt = self.dt
        return np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ])

    def Q(self, x=None):
        dt = self.dt
        q = self.q ** 2
        G = np.array([
            [dt**2/2, 0,       0      ],
            [0,       dt**2/2, 0      ],
            [0,       0,       dt**2/2],
            [dt,      0,       0      ],
            [0,       dt,      0      ],
            [0,       0,       dt     ],
        ])
        return q * G @ G.T

    def h(self, x):
        """Nonlinear observation: Cartesian → (azimuth, elevation)."""
        rel = x[:3] - self.radar_pos
        r = np.linalg.norm(rel)
        theta_az = np.arctan2(rel[1], rel[0])
        theta_el = np.arcsin(rel[2] / (r + 1e-10))
        return np.array([theta_az, theta_el])

    def H(self, x):
        """Jacobian of observation function."""
        rel = x[:3] - self.radar_pos
        px, py, pz = rel
        r = np.linalg.norm(rel)
        r_xy = np.sqrt(px ** 2 + py ** 2)

        # d(theta_az)/d(px,py,pz,vx,vy,vz)
        if r_xy < 1e-10:
            daz_dpx = 0.0
            daz_dpy = 0.0
        else:
            daz_dpx = -py / (r_xy ** 2)
            daz_dpy = px / (r_xy ** 2)

        # d(theta_el)/d(px,py,pz,vx,vy,vz)
        if r < 1e-10:
            del_dpx = 0.0
            del_dpy = 0.0
            del_dpz = 0.0
        else:
            del_dpx = -px * pz / (r ** 2 * r_xy + 1e-10)
            del_dpy = -py * pz / (r ** 2 * r_xy + 1e-10)
            del_dpz = r_xy / (r ** 2 + 1e-10)

        return np.array([
            [daz_dpx, daz_dpy, 0,       0, 0, 0],
            [del_dpx, del_dpy, del_dpz, 0, 0, 0],
        ])

    def R(self, measurement_noise_std=0.02):
        return measurement_noise_std ** 2 * np.eye(2)

    def predict(self, x):
        return self.f(x)

    def observe(self, x):
        return self.h(x)
