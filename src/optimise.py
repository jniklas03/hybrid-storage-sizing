from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.hess import HybridController


def _worker_no_log(args):
    config, x, y, p_bat_init = args

    hc = HybridController.build_controller(
        p_bat_init=p_bat_init,
        logging=False,
        **config,
    )

    hc.simulate(x, y)

    return {
        **config,
        **hc.summary(),
    }

def _worker_log(args):
    config, x, y, p_bat_init = args

    hc = HybridController.build_controller(
        p_bat_init=p_bat_init,
        logging=True,
        **config,
    )
    hc.simulate(x, y)

    return {
        **config,
        "log" : pd.DataFrame(hc.log)
    }

def eval_configs(configs, x, y, p_bat_init, logging, max_workers=None):

    jobs = [
        (config, x, y, p_bat_init)
        for config in configs
    ]

    if logging:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_worker_log, jobs))

        return results
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_worker_no_log, jobs))

        return pd.DataFrame(results)
    

def select_valid(results, error_threshold: float = 0.001):
    valid = results.copy()
    return valid.loc[valid["error_fraction"] <= error_threshold]

def make_front(
    valid_results,
    x_col="n_cells",
    y_col="loss_fraction",
):
    best_per_x = valid_results.loc[
        valid_results.groupby(x_col)[y_col].idxmin()
    ].copy()

    best_per_x = best_per_x.sort_values([x_col, y_col])

    running_best = best_per_x[y_col].cummin()

    front = best_per_x[
        best_per_x[y_col].lt(running_best.shift(fill_value=np.inf))
    ].copy()

    return front