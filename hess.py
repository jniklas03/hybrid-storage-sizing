import math

import numpy as np


class HybridController:

    def __init__(
        self,
        battery_branch,
        supercap_branch,
        tau: float,
        p_bat_init: float = 0.0,
    ):
        # params
        self.battery_branch = battery_branch
        self.supercap_branch = supercap_branch
        self.tau = tau
        self.p_bat_request = p_bat_init

        # log
        self.log = []

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


    def _calc_split(self, p_request, dt):
        alpha = math.exp(-dt / self.tau)

        self.p_bat_request = (
            alpha * self.p_bat_request
            + (1.0 - alpha) * p_request
        )
    
    def step(self, p_request: float, dt: float):
        # split
        self._calc_split(p_request, dt)

        # battery step
        self.battery_branch.step(self.p_bat_request, dt)

        # over-/underflow goes to supercap
        p_bat_actual = self.battery_branch.state.power
        p_sc_request = p_request - p_bat_actual

        # supercap step
        self.supercap_branch.step(p_sc_request, dt)
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
        self.log.append(
            {   
                # battery controller stats
                "p_bat_request": self.p_bat_request,
                "p_bat_actual": p_bat_actual,
                "p_bat_error": self.p_bat_request - p_bat_actual,

                # supercap controller stats
                "p_sc_cmd": p_sc_request,
                "p_sc_actual": p_sc_actual,
                "p_sc_error": p_sc_request - p_sc_actual,

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