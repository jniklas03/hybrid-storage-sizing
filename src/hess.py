import math
from collections.abc import Callable

import numpy as np
from simses.battery import Battery
from simses.battery.cell import CellType
from simses.converter import Converter
from simses.converter.converter import ConverterLossModel
from simses.model.cell.skeleton_SCA import SCA3200
from simses.model.cell.sony_lfp import SonyLFP
from simses.model.converter.fix_efficiency import FixedEfficiency


class HybridController:

    def __init__(
        self,
        battery_branch,
        supercap_branch,
        tau: float,
        p_bat_init: float,
        logging: bool = True
    ):
        # params
        self.battery_branch = battery_branch
        self.supercap_branch = supercap_branch
        self.tau = tau
        self.p_bat_request = p_bat_init

        # log
        self.logging = logging
        self.log = [] if logging else None

        # cumulative losses [Wh]
        self.e_bat_loss = 0.0
        self.e_sc_loss = 0.0
        self.e_bat_conv_loss = 0.0
        self.e_sc_conv_loss = 0.0
        self.e_total_loss = 0.0

        # cumulative throughputs [Wh]
        self.e_bat_thr = 0.0
        self.e_sc_thr = 0.0
        self.e_total_thr = 0.0

        # cumulative error [Wh]
        self.e_total_error = 0.0

        # cumulative request [Wh]
        self.e_request_thr = 0.0

    @classmethod
    def build_controller(
        cls,
        tau: float,
        p_bat_init: float,
        logging: bool = True,
        bat_cell: type[CellType] = SonyLFP,
        sc_cell: type[CellType] = SCA3200,
        start_soc_bat: float = 0.6,
        start_soc_sc: float = 0.6,
        bat_series: int = 16,
        bat_parallel: int = 20,
        sc_series: int = 22,
        sc_parallel: int = 3,
        converter_model: Callable[[], ConverterLossModel] = lambda: FixedEfficiency(1),

    ):
        battery = Battery(
            cell=bat_cell(),
            circuit=(bat_series, bat_parallel),
            initial_states={
                "start_soc": start_soc_bat,
                "start_T": 25.0,
            },
        )
        
        sc = Battery(
            cell=sc_cell(),
            circuit=(sc_series, sc_parallel),
            initial_states={
                "start_soc": start_soc_sc,
                "start_T": 25.0,
            },
        )

        supercap_converter = Converter(
            loss_model=converter_model(),
            max_power=20e3,
            storage=sc,
        )

        battery_converter = Converter(
            loss_model=converter_model(),
            max_power=20e3,
            storage=battery,
        )

        return cls(battery_converter, supercap_converter, tau, p_bat_init, logging)

    def _calc_split(self, p_request, dt):
        alpha = math.exp(-dt / self.tau)

        self.p_bat_request = (
            alpha * self.p_bat_request
            + (1.0 - alpha) * p_request
        )

    @staticmethod
    def _clamp_power(power: float, p_min: float, p_max: float) -> float:
        return max(p_min, min(power, p_max))

    @staticmethod
    def _branch_power_limits(branch, dt: float) -> tuple[float, float]:
        soc_min, soc_max = branch.storage.soc_limits
        p_to_min_soc = branch.target_soc(soc_min, dt)
        p_to_max_soc = branch.target_soc(soc_max, dt)
        return min(p_to_min_soc, p_to_max_soc), max(p_to_min_soc, p_to_max_soc)
    
    def step(self, p_request: float, dt: float):
        # split
        self._calc_split(p_request, dt)

        p_bat_min, p_bat_max = self._branch_power_limits(self.battery_branch, dt)
        p_sc_min, p_sc_max = self._branch_power_limits(self.supercap_branch, dt)

        # Prefer the filtered battery request, then let the supercap cover the
        # residual. If the supercap cannot cover it, move the battery command
        # within its feasible range before accepting an external error.
        p_bat_cmd = self._clamp_power(self.p_bat_request, p_bat_min, p_bat_max)
        p_sc_request = p_request - p_bat_cmd
        p_sc_cmd = self._clamp_power(p_sc_request, p_sc_min, p_sc_max)

        p_bat_request_with_sc_overflow = p_request - p_sc_cmd
        p_bat_cmd = self._clamp_power(p_bat_request_with_sc_overflow, p_bat_min, p_bat_max)
        p_sc_request = p_request - p_bat_cmd
        p_sc_cmd = self._clamp_power(p_sc_request, p_sc_min, p_sc_max)

        # Step each physical branch exactly once with a feasible command.
        self.battery_branch.step(p_bat_cmd, dt)
        p_bat_actual = self.battery_branch.state.power

        self.supercap_branch.step(p_sc_cmd, dt)
        p_sc_actual = self.supercap_branch.state.power

        # system
        p_total_actual = p_bat_actual + p_sc_actual
        p_total_error = p_request - p_total_actual

        # power losses
        p_bat_conv_loss = self.battery_branch.state.loss
        p_sc_conv_loss = self.supercap_branch.state.loss

        p_bat_loss = self.battery_branch.storage.state.loss
        p_sc_loss = self.supercap_branch.storage.state.loss

        p_loss_total = (
            self.battery_branch.state.loss
            + self.battery_branch.storage.state.loss
            + self.supercap_branch.state.loss
            + self.supercap_branch.storage.state.loss
        )

        # energy losses
        dE_bat_loss = p_bat_loss * dt / 3600
        self.e_bat_loss += dE_bat_loss

        dE_sc_loss = p_sc_loss * dt / 3600
        self.e_sc_loss += dE_sc_loss

        dE_bat_conv_loss = p_bat_conv_loss * dt / 3600
        self.e_bat_conv_loss += dE_bat_conv_loss

        dE_sc_conv_loss = p_sc_conv_loss * dt / 3600
        self.e_sc_conv_loss += dE_sc_conv_loss

        dE_total_loss = p_loss_total * dt / 3600
        self.e_total_loss += dE_total_loss

        # throughput
        dE_bat_thr = abs(p_bat_actual) * dt / 3600
        self.e_bat_thr += dE_bat_thr

        dE_sc_thr = abs(p_sc_actual) * dt / 3600
        self.e_sc_thr += dE_sc_thr

        dE_total_thr = abs(p_total_actual) * dt / 3600
        self.e_total_thr += dE_total_thr

        dE_request_thr = abs(p_request) * dt / 3600
        self.e_request_thr += dE_request_thr

        # error
        dE_total_error = abs(p_total_error) * dt / 3600
        self.e_total_error += dE_total_error

        # cumulative (!) loss fraction
        loss_fraction = (
            self.e_total_loss / self.e_total_thr if self.e_total_thr > 0 else np.nan
        )

        # cumulative (!) error fraction
        error_fraction = (
            self.e_total_error / self.e_request_thr if self.e_request_thr > 0 else np.nan
        )

        # logging
        if self.logging:
            self.log.append(
                {   
                    # battery controller stats
                    "p_bat_request": self.p_bat_request,
                    "p_bat_cmd": p_bat_cmd,
                    "p_bat_min": p_bat_min,
                    "p_bat_max": p_bat_max,
                    "p_bat_actual": p_bat_actual,
                    "p_bat_error": p_bat_cmd - p_bat_actual,

                    # supercap controller stats
                    "p_sc_request": p_sc_request,
                    "p_sc_cmd": p_sc_cmd,
                    "p_sc_min": p_sc_min,
                    "p_sc_max": p_sc_max,
                    "p_sc_actual": p_sc_actual,
                    "p_sc_error": p_sc_cmd - p_sc_actual,

                    # total controller stats
                    "p_total_request": p_request,
                    "p_total_actual": p_total_actual,
                    "p_total_error": p_total_error,

                    # losses
                    "p_bat_loss": p_bat_loss,
                    "p_sc_loss": p_sc_loss,
                    "p_bat_conv_loss": p_bat_conv_loss,
                    "p_sc_conv_loss": p_sc_conv_loss,
                    "p_loss_total": p_loss_total,

                    "dE_bat_loss": dE_bat_loss,
                    "dE_sc_loss": dE_sc_loss,
                    "dE_bat_conv_loss": dE_bat_conv_loss,
                    "dE_sc_conv_loss": dE_sc_conv_loss,
                    "dE_total_loss": dE_total_loss,

                    "e_bat_loss": self.e_bat_loss,
                    "e_sc_loss": self.e_sc_loss,
                    "e_bat_conv_loss": self.e_bat_conv_loss,
                    "e_sc_conv_loss": self.e_sc_conv_loss,
                    "e_total_loss": self.e_total_loss,

                    # throughput
                    "dE_bat_thr": dE_bat_thr,
                    "dE_sc_thr": dE_sc_thr,
                    "dE_total_thr": dE_total_thr,

                    "e_bat_thr": self.e_bat_thr,
                    "e_sc_thr": self.e_sc_thr,
                    "e_total_thr": self.e_total_thr,

                    # total errors
                    "dE_total_error": dE_total_error,
                    "e_total_error": self.e_total_error,

                    # fractions / efficiency metrics
                    "loss_fraction": loss_fraction,
                    "error_fraction" : error_fraction,

                    # storage states
                    "bat_v" : self.battery_branch.storage.state.v,
                    "bat_i" : self.battery_branch.storage.state.i,
                    "bat_soc" : self.battery_branch.storage.state.soc,

                    "sc_v" : self.supercap_branch.storage.state.v,
                    "sc_i" : self.supercap_branch.storage.state.i,
                    "sc_soc" : self.supercap_branch.storage.state.soc,
                }
        )

    def simulate(self, x, y):
        for time, power in zip(x,y):
            self.step(power, time)

    def summary(self):
        return {
            "loss_fraction": (
                self.e_total_loss / self.e_total_thr
                if self.e_total_thr > 0
                else np.nan
            ),
            "error_fraction": (
                self.e_total_error / self.e_request_thr
                if self.e_request_thr > 0
                else np.nan
            ),
            "e_total_loss": self.e_total_loss,
            "e_total_error": self.e_total_error,
            "e_bat_thr": self.e_bat_thr,
            "e_sc_thr": self.e_sc_thr,
        }
