"""Live viewer and a convenient run() helper."""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

import math
import time as _time
from collections import deque
from collections.abc import Callable

from .core import CartPole, Sensors


class Result:
    """What run() gives back: the recorded data and a plot method."""

    def __init__(self, sim: CartPole) -> None:
        self.time = sim.history["time"]
        self.angle = sim.history["angle"]
        self.position = sim.history["position"]
        self.motor = sim.history["motor"]
        self.hit_wall = sim.hit_wall

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        axes[0].plot(self.time, [math.degrees(a) for a in self.angle])
        axes[0].set_ylabel("angle (deg)")
        axes[1].plot(self.time, self.position)
        axes[1].set_ylabel("position (m)")
        axes[2].plot(self.time, self.motor)
        axes[2].set_ylabel("motor command")
        axes[2].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()


def run(
    controller: Callable[[Sensors], float],
    sim: CartPole | None = None,
    duration: float = 20.0,
    visualize: bool = True,
    stop_on_fall: bool = True,
    speed: float = 1.0,
) -> Result:
    """Run a controller on the simulation.

    controller : function taking a Sensors tuple (time, angle, position) and
                 returning a motor command between -1 and 1. It is called once
                 per control step, so it may keep its own memory (e.g. a class
                 with __call__, or a closure) for integral/derivative terms.
    sim        : a CartPole; a default one is created if omitted.
    duration   : simulated seconds to run.
    visualize  : True opens a live window (arrow keys nudge the pendulum);
                 False runs as fast as possible with no window.
    speed      : 1.0 = real time, 0.5 = slow motion (visualize only).

    Returns a Result with the recorded data (call .plot() on it).
    """
    sim = sim or CartPole()

    def control_step() -> None:
        sim.set_motor(controller(sim.read_sensors()))
        sim.step()

    if not visualize:
        while sim.time < duration:
            control_step()
        return Result(sim)

    _animate(sim, control_step, duration, stop_on_fall, speed)
    return Result(sim)


def _animate(
    sim: CartPole,
    control_step: Callable[[], None],
    duration: float,
    stop_on_fall: bool,
    speed: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.backend_bases import KeyEvent
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.text import Text

    p = sim.params
    L = p.rail_half_length
    pole_len = 2 * p.pole_half_length
    cart_w, cart_h = 0.22, 0.12

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes((0.06, 0.40, 0.88, 0.55))
    ax.set_xlim(-L - 0.3, L + 0.3)
    ax.set_ylim(-0.25, pole_len + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-L - 0.1, L + 0.1], [0, 0], color="0.3", lw=3, zorder=1)
    for sx in (-L, L):  # end stops, drawn where the cart's centre would stop
        ax.add_patch(
            Rectangle(
                (sx - 0.02 + (cart_w / 2 if sx > 0 else -cart_w / 2), 0),
                0.04,
                0.2,
                color="firebrick",
                zorder=1,
            )
        )
    ax.plot([0, 0], [-0.05, 0.05], color="0.6", lw=1)  # centre mark

    cart = Rectangle((0, 0), cart_w, cart_h, color="steelblue", zorder=3)
    ax.add_patch(cart)
    (pole,) = ax.plot([], [], color="0.15", lw=5, solid_capstyle="round", zorder=2)
    bob = Circle((0, 0), 0.04, color="darkorange", zorder=4)
    ax.add_patch(bob)
    status = ax.text(0, pole_len + 0.25, "", ha="center", fontsize=11)
    ax.text(
        0,
        -0.2,
        "arrow keys: nudge the pendulum    |    close window to stop",
        ha="center",
        fontsize=9,
        color="0.4",
    )

    # strip charts
    window = 6.0
    ax_a = fig.add_axes((0.10, 0.22, 0.85, 0.14))
    ax_m = fig.add_axes((0.10, 0.05, 0.85, 0.14), sharex=ax_a)
    ax_a.set_ylabel("angle (deg)")
    ax_m.set_ylabel("motor")
    ax_m.set_xlabel("time (s)")
    ax_m.set_ylim(-1.1, 1.1)
    ax_a.set_ylim(-35, 35)
    ax_a.axhline(0, color="0.7", lw=0.8)
    ax_m.axhline(0, color="0.7", lw=0.8)
    ax_a.grid(alpha=0.3)
    ax_m.grid(alpha=0.3)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    (line_a,) = ax_a.plot([], [], color="darkorange")
    (line_m,) = ax_m.plot([], [], color="steelblue")
    ts, angs, cmds = deque(), deque(), deque()

    def on_key(event: KeyEvent) -> None:
        if event.key == "left":
            sim.nudge(-0.5)
        elif event.key == "right":
            sim.nudge(0.5)

    fig.canvas.mpl_connect("key_press_event", on_key)

    clock = {"start": None, "stopped": False}

    def draw_frame(_: int) -> tuple[Rectangle, Line2D, Circle, Text, Line2D, Line2D]:
        if clock["start"] is None:
            clock["start"] = _time.perf_counter() - sim.time / speed
        target = (_time.perf_counter() - clock["start"]) * speed
        steps = 0
        done = sim.time >= duration
        while not done and sim.time < target and steps < 30:
            control_step()
            steps += 1
            done = sim.time >= duration
            ts.append(sim.time)
            angs.append(math.degrees(sim.true_state()[2]))
            cmds.append(sim.history["motor"][-1])
        while ts and ts[0] < sim.time - window:
            ts.popleft()
            angs.popleft()
            cmds.popleft()

        state = sim.true_state()
        x, th = state[0], state[2]
        cart.set_xy((x - cart_w / 2, 0))
        px, py = x, cart_h
        tx, ty = px + pole_len * math.sin(th), py + pole_len * math.cos(th)
        pole.set_data([px, tx], [py, ty])
        bob.center = (tx, ty)
        msg = f"t = {sim.time:5.2f} s     angle = {math.degrees(th):6.1f}°     position = {x:5.2f} m"
        if sim.time >= duration:
            msg += "\nDone."
        status.set_text(msg)
        line_a.set_data(ts, angs)
        line_m.set_data(ts, cmds)
        ax_a.set_xlim(max(0, sim.time - window), max(window, sim.time))
        return cart, pole, bob, status, line_a, line_m

    anim = FuncAnimation(
        fig, draw_frame, interval=20, blit=False, cache_frame_data=False
    )
    fig._anim = anim  # type: ignore[attr-defined]  # keep a reference alive
    plt.show()
