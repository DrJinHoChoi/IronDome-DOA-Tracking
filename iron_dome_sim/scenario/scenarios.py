"""Pre-defined simulation scenarios.

Provides ready-to-use scenario configurations for Iron Dome simulation.
"""

import numpy as np
from ..signal_model.array import UniformLinearArray
from .radar_network import RadarSite, RadarNetwork
from .threat_generator import ThreatGenerator
from .interceptor import Interceptor


def small_scenario():
    """Small test scenario: 8 sensors, 15 threats, 2 radars.

    Good for quick testing and debugging.
    """
    # Radar sites
    radar1 = RadarSite(
        position=[0, 0, 50],
        array=UniformLinearArray(M=8, d=0.5),
        name="Radar-Alpha",
        max_range=30000,
    )
    radar2 = RadarSite(
        position=[5000, 3000, 50],
        array=UniformLinearArray(M=8, d=0.5),
        name="Radar-Bravo",
        max_range=30000,
    )
    network = RadarNetwork([radar1, radar2], snr_db=10)

    # Threats
    gen = ThreatGenerator(dt=0.1)
    launch_sites = [
        [-20000, 5000, 0],
        [-15000, -3000, 0],
    ]
    threats = gen.generate_salvo(
        launch_sites=launch_sites,
        target_area_center=[0, 0, 0],
        target_area_radius=3000,
        num_threats=15,
        salvo_type="staggered",
        time_spread=3.0,
    )

    # Protected areas
    protected = [
        ([0, 0, 0], 5000),      # City center
        ([2000, 1000, 0], 2000), # Industrial zone
    ]
    interceptor = Interceptor(protected)

    return {
        'network': network,
        'threats': threats,
        'threat_gen': gen,
        'interceptor': interceptor,
        'dt': 0.1,
        'duration': 60,  # seconds
        'name': 'Small Scenario (15 threats, 2 radars)',
    }


def iron_dome_scenario():
    """Full Iron Dome scenario: 8 sensors, 50+ threats, 3 radars.

    Realistic multi-radar network defending against a mass salvo attack.
    Demonstrates the advantage of underdetermined DOA estimation
    (50+ targets > 8 sensors).
    """
    # Radar network (3 sites in triangular formation)
    radar1 = RadarSite(
        position=[0, 0, 100],
        array=UniformLinearArray(M=8, d=0.5),
        name="Radar-Alpha",
        max_range=50000,
    )
    radar2 = RadarSite(
        position=[8000, 5000, 80],
        array=UniformLinearArray(M=8, d=0.5),
        name="Radar-Bravo",
        max_range=50000,
    )
    radar3 = RadarSite(
        position=[4000, -6000, 120],
        array=UniformLinearArray(M=8, d=0.5),
        name="Radar-Charlie",
        max_range=50000,
    )
    network = RadarNetwork([radar1, radar2, radar3], snr_db=12)

    # Multi-salvo attack from 4 launch sites
    gen = ThreatGenerator(dt=0.1, drag_coefficient=0.001)
    launch_sites = [
        [-30000, 10000, 0],    # North-West
        [-25000, -8000, 0],    # South-West
        [-35000, 0, 0],        # West
        [-20000, 15000, 0],    # North
    ]

    # Mixed threat types
    missile_types = (
        ["short_range"] * 30 +
        ["medium_range"] * 15 +
        ["cruise"] * 10
    )

    threats = gen.generate_salvo(
        launch_sites=launch_sites,
        target_area_center=[3000, 0, 0],
        target_area_radius=5000,
        num_threats=55,
        salvo_type="random",
        missile_types=missile_types,
        time_spread=8.0,
    )

    # Protected areas (multiple zones)
    protected = [
        ([0, 0, 0], 5000),        # Primary city
        ([3000, 2000, 0], 3000),   # Residential area
        ([5000, -1000, 0], 2000),  # Military base
        ([1000, 4000, 0], 1500),   # Hospital
    ]
    interceptor = Interceptor(
        protected,
        intercept_range=20000,
        intercept_prob_base=0.9,
        max_simultaneous=15,
    )

    return {
        'network': network,
        'threats': threats,
        'threat_gen': gen,
        'interceptor': interceptor,
        'dt': 0.1,
        'duration': 120,  # seconds
        'name': 'Iron Dome Full Scenario (55 threats, 3 radars)',
    }
