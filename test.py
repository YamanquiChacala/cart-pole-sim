from cart_pole_sim import CartPole, Sensors, run


def controller(s: Sensors):
    return 0.2  # <- your code here


if __name__ == "__main__":
    sim = CartPole(initial_angle=0.1)
    result = run(controller, sim, duration=20)
    result.plot()
