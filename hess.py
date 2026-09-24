import math


class HybridController:

    def __init__(
        self,
        battery_branch,
        supercap_branch,
        tau: float,
        p_slow_init: float = 0.0,
    ):
        self.battery_branch = battery_branch
        self.supercap_branch = supercap_branch
        self.tau = tau
        self._p_slow = p_slow_init

    def step(self, p_request: float, dt: float):

        # slow component -> battery
        alpha = math.exp(-dt / self.tau)

        self._p_slow = (
            alpha * self._p_slow
            + (1.0 - alpha) * p_request
        )

        self.battery_branch.step(self._p_slow, dt)

        p_battery = self.battery_branch.state.power
        p_supercap = p_request - p_battery
        
        self.supercap_branch.step(p_supercap, dt)