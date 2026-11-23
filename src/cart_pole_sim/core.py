"""Physics of a pendulum on a motorised cart that moves along a 1-D rail.

Conventions
-----------
* angle    : radians, 0 = pendulum perfectly upright,
             positive = leaning to the RIGHT (towards +position).
* position : metres, 0 = centre of the rail, positive = right.
* motor    : a command between -1 and +1.
             +1 = full power pushing the cart right, -1 = full power left.
"""

import math
import random
from dataclasses import astuple, dataclass, fields
from typing import NamedTuple


class Sensors(NamedTuple):
    """Data for the controller

    Attribuest
        time: seconds since the start of simulation
        angle: pendulum angle in radians, 0 upgight, positive, leaning to the rigth
        position: cart's position in meters. 0 - center, positive, to the rigth
    """

    time: float
    angle: float
    position: float


@dataclass(slots=True)
class Params:
    """Physical parameters.

    Change to make the simulation harder
    """

    cart_mass = 0.8  # Kg
    pole_mass = 0.1  # Kg
    pole_half_length = 0.25  # m (from pivot to center of mass)
    gravity = 9.81  # m/s^2
    max_force = 12.0  # N produced by the motor at command = 1
    motor_lag = 0.02  # s, motor takes time to reach the commanded force
    cart_friction = 0.5  # N per (m/s)
    pivot_friction = 0.002  # N*m per (rad/s)
    rail_half_length = 1.0  # m, the cart hits the walls here.
    wall_restitution = 0.3  # 0 = cart stops dead, 1 = perfect elastic bounce


@dataclass(slots=True)
class _State:
    """Mutable state of the simulation."""

    position: float  # m, 0=center, position of the cart
    velocity: float  # m/s, cart speed
    angle: float  # rad, 0=upright, pole angle
    ang_velocity: float  # rad/s, pole angular velocity
    motor_force: float  # N, force being applied by the motor

    def advanced(self, d: _Derivative, dt: float) -> _State:
        """State reached by moving for `dt` seconds along the slope of `d`."""
        return _State(
            position=self.position + d.d_position * dt,
            velocity=self.velocity + d.d_velocity * dt,
            angle=self.angle + d.d_angle * dt,
            ang_velocity=self.ang_velocity + d.d_ang_velocity * dt,
            motor_force=self.motor_force + d.d_motor_force * dt,
        )

    def lerp(self, other: _State, t: float) -> _State:
        """Linearly interpolate between this state (t=0) and `other` (t=1)"""
        return _State(
            **{
                f.name: getattr(self, f.name)
                + (getattr(other, f.name) - getattr(self, f.name)) * t
                for f in fields(self)
            }
        )


@dataclass(slots=True)
class _Derivative:
    """Time derivation of `_State`."""

    d_position: float  # m/s, same as _State's velocity
    d_velocity: float  # m/s^2, cart's acceleration
    d_angle: float  # rad/s, same as _State's pole angle
    d_ang_velocity: float  # rad/s^2, pole angular acceleration
    d_motor_force: float  # N/s, change in motor force

    @staticmethod
    def rk4_average(
        k1: _Derivative, k2: _Derivative, k3: _Derivative, k4: _Derivative
    ) -> _Derivative:
        """The RK4 weighted mean slope: (k1 + 2*k2 + 2*k3 + k4) / 6."""
        return _Derivative(
            *(
                (a + 2 * b + 2 * c + d) / 6
                for a, b, c, d in zip(
                    astuple(k1), astuple(k2), astuple(k3), astuple(k4)
                )
            )
        )


class CartPole:
    """The simulated hardware.

    Typical use (the control loop is yours to write)::

        sim = CartPole(initial_angle=0.1)
        s = sim.read_sensors()
        sim.set_motor(my_controller(s))
        sim.step()
    """

    def __init__(
        self,
        initial_angle: float = 0.5,
        initial_position: float = 0,
        control_dt: float = 0.01,
        params: Params | None = None,
        angle_noise: float = 0,
        position_noise: float = 0,
        seed: int | None = None,
    ) -> None:
        """
        control_dt     : seconds between controller updates (0.01 = 100 Hz)
        angle_noise    : standard deviation of sensor noise in radians (try 0.002)
        position_noise : standard deviation of sensor noise in metres (try 0.001)
        """
        self.params = params or Params()
        self.control_dt = control_dt
        self.angle_noise = angle_noise
        self.position_noise = position_noise
        self._rng = random.Random(seed)
        self._initial = (initial_angle, initial_position)
        self._phys_dt = 0.001
        self.reset()

    def reset(self):
        """Returns the simulation to it's starting position."""
        angle, position = self._initial
        self._state = _State(position, 0, angle, 0, 0)
        self._command = 0.0
        self.time = 0
        self.hit_wall = False
        self.history: dict[str, list[float]] = {
            "time": [],
            "angle": [],
            "position": [],
            "motor": [],
        }

    def read_sensors(self) -> Sensors:
        """Reads the sensors, with noise, if enabled"""
        ang = self._state.angle
        pos = self._state.position
        if self.angle_noise:
            ang += self._rng.gauss(0, self.angle_noise)
        if self.position_noise:
            pos += self._rng.gauss(0, self.position_noise)
        return Sensors(self.time, ang, pos)

    def true_state(self) -> tuple[float, float, float, float, float]:
        return astuple(self._state)

    def set_motor(self, command: float):
        """Set the motor command, clipped to [-1, 1]."""
        self._command = max(-1.0, min(1.0, command))

    def nudge(self, strength: float = 1.5) -> None:
        """Hit the pendulum. strength is the angular velocity (rad/s) it
        suddenly gains; positive pushes it to the right."""
        self._state.ang_velocity += strength

    def step(self) -> None:
        """Advance the simulation by control_dt seconds."""
        s = self._state

        self.history["time"].append(self.time)
        self.history["angle"].append(s.angle)
        self.history["position"].append(s.position)
        self.history["motor"].append(self._command)

        n = max(1, round(self.control_dt / self._phys_dt))
        h = self.control_dt / n
        for _ in range(n):
            self.physics_step(h)

        self.time += self.control_dt

    def _derivatives(self, s: _State, command: float) -> _Derivative:
        """Time derivative of the state."""
        p = self.params
        M, m, l, g = p.cart_mass, p.pole_mass, p.pole_half_length, p.gravity
        sin_a, cos_a = math.sin(s.angle), math.cos(s.angle)

        # Total horizontal force on the cart: motor minus viscous friction
        force = s.motor_force - p.cart_friction * s.velocity

        # Equations of motion as a 2x2 linear system in (x'', theta''):
        #   a11*x'' + a12*theta'' = b1
        #   a12*x'' + a22*theta'' = b2
        a11 = M + m
        a12 = m * l * cos_a
        a22 = (4.0 / 3.0) * m * l * l
        b1 = force + m * l * sin_a * s.ang_velocity**2
        b2 = m * g * l * sin_a - p.pivot_friction * s.ang_velocity

        det = a11 * a22 - a12 * a12
        acc = (b1 * a22 - a12 * b2) / det
        ang_acc = (a11 * b2 - a12 * b1) / det

        # First-order motor lag: force relaxes towards max_force * command
        motor_rate = (p.max_force * command - s.motor_force) / p.motor_lag

        return _Derivative(
            d_position=s.velocity,
            d_velocity=acc,
            d_angle=s.ang_velocity,
            d_ang_velocity=ang_acc,
            d_motor_force=motor_rate,
        )

    def _rk4(self, s: _State, h: float) -> _State:
        """One classic Runge-Kutta 4 step of length h."""
        c = self._command
        k1 = self._derivatives(s, c)
        k2 = self._derivatives(s.advanced(k1, h / 2), c)
        k3 = self._derivatives(s.advanced(k2, h / 2), c)
        k4 = self._derivatives(s.advanced(k3, h), c)
        return s.advanced(_Derivative.rk4_average(k1, k2, k3, k4), h)

    def _bounce(self, s: _State, wall_side: float) -> _State:
        """Resolve an instantaneous collision of the cart with a wall.
        wall_side is +1 for the right wall, -1 for the left."""
        if s.velocity * wall_side <= 0:  # already moving away, nothing to do
            return s
        p = self.params
        m, l = p.pole_mass, p.pole_half_length
        # Tiny impacts are treated as inelastic so the cart doesn't chatter
        e = p.wall_restitution if abs(s.velocity) > 0.05 else 0.0

        delta_v = -(1.0 + e) * s.velocity
        # The wall's impulse acts on the cart only, so the pole feels it through
        # the pivot. Same mass matrix as above, with zero external torque:
        #   a12*dv + a22*dw = 0  ->  dw = -(a12/a22) * dv
        a12 = m * l * math.cos(s.angle)
        a22 = (4.0 / 3.0) * m * l * l
        s.velocity += delta_v
        s.ang_velocity -= (a12 / a22) * delta_v
        return s

    def physics_step(self, h: float):
        old = self._state
        new = self._rk4(old, h)

        limit = self.params.rail_half_length
        if abs(new.position) > limit:
            self.hit_wall = True
            side = math.copysign(1, new.position)
            wall = side * limit

            # Fraction of the sub-step at which the cart reached the wall
            t = (wall - old.position) / (new.position - old.position)
            t = max(0.0, min(1.0, t))

            impact = old.lerp(new, t)
            impact.position = wall
            impact = self._bounce(impact, side)

            # Finish the remainder of the sub-step from the post-bounce state
            new = self._rk4(impact, (1.0 - t) * h) if t < 1.0 else impact
            new.position = max(-limit, min(limit, new.position))  # safety clamp

        self._state = new
