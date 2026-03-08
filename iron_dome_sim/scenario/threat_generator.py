"""Threat (missile/rocket) trajectory generation.

Generates realistic 3D ballistic trajectories for Iron Dome simulation.
Supports multiple launch sites, salvo patterns, and missile types.
"""

import numpy as np


class Threat:
    """Single threat (missile/rocket) with ballistic trajectory.

    Attributes:
        id: Unique identifier.
        launch_pos: Launch position [x, y, z] in meters.
        target_pos: Intended impact position [x, y, z].
        missile_type: "short_range", "medium_range", or "cruise".
        trajectory: Array of positions over time, shape (N_steps, 3).
        velocities: Array of velocities over time, shape (N_steps, 3).
        is_active: Whether threat is still in flight.
    """

    _next_id = 0

    def __init__(self, launch_pos, target_pos, launch_time=0.0,
                 missile_type="short_range"):
        self.id = Threat._next_id
        Threat._next_id += 1
        self.launch_pos = np.array(launch_pos, dtype=float)
        self.target_pos = np.array(target_pos, dtype=float)
        self.launch_time = launch_time
        self.missile_type = missile_type
        self.trajectory = []
        self.velocities = []
        self.is_active = True
        self.is_intercepted = False
        self.impact_time = None

    @property
    def current_pos(self):
        if len(self.trajectory) > 0:
            return self.trajectory[-1]
        return self.launch_pos.copy()


class ThreatGenerator:
    """Generates multiple threat trajectories for simulation.

    Args:
        dt: Time step in seconds.
        gravity: Gravitational acceleration (m/s^2).
        drag_coefficient: Aerodynamic drag coefficient (0 = no drag).
    """

    MISSILE_PARAMS = {
        "short_range": {"speed": 200, "max_alt": 5000, "range": 10000},
        "medium_range": {"speed": 500, "max_alt": 15000, "range": 40000},
        "cruise": {"speed": 250, "max_alt": 1000, "range": 30000},
    }

    def __init__(self, dt=0.1, gravity=9.81, drag_coefficient=0.0):
        self.dt = dt
        self.g = gravity
        self.Cd = drag_coefficient

    def generate_salvo(self, launch_sites, target_area_center, target_area_radius,
                       num_threats=50, salvo_type="simultaneous",
                       missile_types=None, time_spread=5.0):
        """Generate a salvo of threats from multiple launch sites.

        Args:
            launch_sites: List of launch positions [x, y, z].
            target_area_center: Center of target area [x, y, z].
            target_area_radius: Radius of target dispersion (meters).
            num_threats: Total number of threats to generate.
            salvo_type: "simultaneous", "staggered", or "random".
            missile_types: List of missile types (or None for mixed).
            time_spread: Time spread for staggered/random launches (seconds).

        Returns:
            List of Threat objects with computed trajectories.
        """
        threats = []

        if missile_types is None:
            missile_types = ["short_range"] * (num_threats // 2) + \
                           ["medium_range"] * (num_threats - num_threats // 2)
            np.random.shuffle(missile_types)

        for i in range(num_threats):
            # Select launch site (round-robin)
            site = launch_sites[i % len(launch_sites)]
            launch_pos = np.array(site, dtype=float)
            # Add small random offset to launch position
            launch_pos[:2] += np.random.randn(2) * 100

            # Random target point within target area
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, target_area_radius)
            target_pos = np.array(target_area_center, dtype=float)
            target_pos[0] += radius * np.cos(angle)
            target_pos[1] += radius * np.sin(angle)

            # Launch time
            if salvo_type == "simultaneous":
                launch_time = 0.0
            elif salvo_type == "staggered":
                launch_time = i * time_spread / num_threats
            else:  # random
                launch_time = np.random.uniform(0, time_spread)

            mtype = missile_types[i % len(missile_types)]
            threat = Threat(launch_pos, target_pos, launch_time, mtype)
            self._compute_trajectory(threat)
            threats.append(threat)

        return threats

    def _compute_trajectory(self, threat):
        """Compute full ballistic trajectory for a threat.

        Uses projectile motion with optional drag.
        """
        params = self.MISSILE_PARAMS[threat.missile_type]

        launch = threat.launch_pos
        target = threat.target_pos

        # Horizontal distance and direction
        dx = target[0] - launch[0]
        dy = target[1] - launch[1]
        horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
        direction = np.array([dx, dy]) / (horizontal_dist + 1e-10)

        if threat.missile_type == "cruise":
            # Cruise missile: nearly level flight at low altitude
            self._compute_cruise_trajectory(threat, direction, horizontal_dist, params)
        else:
            # Ballistic trajectory
            self._compute_ballistic_trajectory(threat, direction, horizontal_dist, params)

    def _compute_ballistic_trajectory(self, threat, direction, horiz_dist, params):
        """Compute ballistic (parabolic) trajectory."""
        speed = params["speed"]
        max_alt = params["max_alt"]

        # Compute launch angle for desired range
        # For ballistic: range = v^2 sin(2*theta) / g
        sin2theta = min(1.0, self.g * horiz_dist / (speed ** 2 + 1e-10))
        launch_angle = 0.5 * np.arcsin(sin2theta)

        # Adjust for desired altitude
        launch_angle = max(launch_angle, np.radians(30))
        launch_angle = min(launch_angle, np.radians(70))

        # Initial velocity components
        v_horiz = speed * np.cos(launch_angle)
        v_vert = speed * np.sin(launch_angle)
        vx = v_horiz * direction[0]
        vy = v_horiz * direction[1]
        vz = v_vert

        # Simulate trajectory
        pos = threat.launch_pos.copy()
        vel = np.array([vx, vy, vz])

        threat.trajectory = [pos.copy()]
        threat.velocities = [vel.copy()]

        max_steps = int(200 / self.dt)  # Max 200 seconds
        for step in range(max_steps):
            # Drag force
            if self.Cd > 0:
                speed_curr = np.linalg.norm(vel)
                drag = -0.5 * self.Cd * speed_curr * vel
            else:
                drag = np.zeros(3)

            # Update velocity (gravity + drag)
            vel = vel + np.array([drag[0], drag[1], -self.g + drag[2]]) * self.dt

            # Update position
            pos = pos + vel * self.dt

            threat.trajectory.append(pos.copy())
            threat.velocities.append(vel.copy())

            # Check ground impact
            if pos[2] <= 0 and step > 0:
                threat.is_active = False
                threat.impact_time = (step + 1) * self.dt + threat.launch_time
                break

        threat.trajectory = np.array(threat.trajectory)
        threat.velocities = np.array(threat.velocities)

    def _compute_cruise_trajectory(self, threat, direction, horiz_dist, params):
        """Compute cruise missile trajectory (low altitude, nearly level)."""
        speed = params["speed"]
        cruise_alt = params["max_alt"]

        # Phase 1: Climb to cruise altitude
        # Phase 2: Level flight toward target
        # Phase 3: Terminal dive

        pos = threat.launch_pos.copy()
        threat.trajectory = [pos.copy()]
        threat.velocities = []

        # Climb phase
        climb_angle = np.radians(15)
        v_climb = np.array([
            speed * np.cos(climb_angle) * direction[0],
            speed * np.cos(climb_angle) * direction[1],
            speed * np.sin(climb_angle),
        ])

        while pos[2] < cruise_alt:
            pos = pos + v_climb * self.dt
            threat.trajectory.append(pos.copy())
            threat.velocities.append(v_climb.copy())

        # Cruise phase
        v_cruise = np.array([speed * direction[0], speed * direction[1], 0.0])
        remaining_dist = np.linalg.norm(threat.target_pos[:2] - pos[:2])
        dive_dist = cruise_alt / np.tan(np.radians(30))

        while remaining_dist > dive_dist:
            pos = pos + v_cruise * self.dt
            threat.trajectory.append(pos.copy())
            threat.velocities.append(v_cruise.copy())
            remaining_dist = np.linalg.norm(threat.target_pos[:2] - pos[:2])

        # Terminal dive
        dive_angle = np.radians(30)
        v_dive = np.array([
            speed * np.cos(dive_angle) * direction[0],
            speed * np.cos(dive_angle) * direction[1],
            -speed * np.sin(dive_angle),
        ])

        max_steps = int(100 / self.dt)
        for _ in range(max_steps):
            pos = pos + v_dive * self.dt
            threat.trajectory.append(pos.copy())
            threat.velocities.append(v_dive.copy())
            if pos[2] <= 0:
                threat.is_active = False
                break

        threat.trajectory = np.array(threat.trajectory)
        threat.velocities = np.array(threat.velocities)

    def get_positions_at_time(self, threats, t):
        """Get all threat positions at a specific time.

        Args:
            threats: List of Threat objects.
            t: Time in seconds.

        Returns:
            positions: Array of active threat positions, shape (K_active, 3).
            active_ids: List of active threat IDs.
        """
        positions = []
        active_ids = []

        for threat in threats:
            if t < threat.launch_time:
                continue

            t_rel = t - threat.launch_time
            step = int(t_rel / self.dt)

            if step < len(threat.trajectory) and threat.trajectory[step][2] > 0:
                positions.append(threat.trajectory[step])
                active_ids.append(threat.id)

        if len(positions) == 0:
            return np.empty((0, 3)), []

        return np.array(positions), active_ids
