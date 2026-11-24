"""Worked solution (for the teacher): two cascaded PID loops.

Inner loop : keeps the pendulum at the angle the outer loop asks for.
Outer loop : if the cart is off-centre, ask the inner loop to lean the
             pendulum *towards the centre*, which makes the cart drive back.
"""

from cart_pole_sim import CartPole, Sensors, run


class PID:
    """PID with the derivative taken on the *measurement* (not the error).

    Why? If the setpoint changes suddenly (or comes from another PID, as the
    angle setpoint does here), differentiating the error gives a huge spike
    ("derivative kick"). Differentiating the measurement avoids that.

    Note the sign: error = measurement - setpoint. Pushing the cart right
    makes the pendulum lean LEFT, so to fix a rightward lean the motor must
    push right, i.e. positive output for positive (measurement - setpoint).
    """

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev: tuple[float, float] | None = None  # (time, measurement)

    def update(self, setpoint: float, measurement: float, t: float):
        error = measurement - setpoint
        if self.prev is None:
            dt, derivative = 0.0, 0.0
        else:
            prev_time, prev_measurement = self.prev
            dt = t - prev_time
            derivative = (measurement - prev_measurement) / dt
        self.integral += error * dt
        self.prev = (t, measurement)
        return self.kp * error + self.ki * self.integral + self.kd * derivative


angle_pid = PID(kp=3.0, ki=0.0, kd=0.3)
position_pid = PID(kp=0.1, ki=0.0, kd=0.15)


def controller(s: Sensors) -> float:
    # Off-centre to the right -> ask for a lean to the left (and vice versa).
    wanted_angle = -position_pid.update(0.0, s.position, s.time)
    return angle_pid.update(wanted_angle, s.angle, s.time)


if __name__ == "__main__":
    sim = CartPole(initial_angle=0.5)
    result = run(controller, sim, duration=30)
    result.plot()
