import matplotlib.pyplot as plt


def plot_simulation(
        results,
        x_axis,
        idx,

):
    plt.figure(figsize=(14,7))
    plt.plot(
        x_axis,
        results[idx]["log"]["dE_total_thr"],
        label="Requested throughput"
    )
    plt.plot(
        x_axis,
        results[idx]["log"]["dE_bat_thr"],
        label="Battery throughput"
    )
    plt.plot(
        x_axis,
        results[idx]["log"]["dE_sc_thr"],
        label="Supercap throughput"
    )
    plt.plot(
        x_axis,
        results[idx]["log"]["dE_total_loss"],
        label="Total loss [%]"
    )
    plt.plot(
        x_axis,
        results[idx]["log"]["dE_total_error"],
        label="Total error [%]"
    )
    plt.xlabel("t [min]")
    plt.ylabel("Energy [kWh]")

    plt.legend()

    plt.figure(figsize=(14,7))
    plt.plot(
        x_axis,
        results[idx]["log"]["p_total_actual"],
        label="Requested deliverable power"
    )
    plt.plot(
        x_axis,
        results[idx]["log"]["p_bat_actual"],
        label="Battery power"
    )
    plt.plot(
        x_axis,
        results[idx]["log"]["p_sc_actual"],
        label="Supercap power"
    )
    plt.xlabel("t [min]")
    plt.ylabel("Power [W]")

    plt.legend()

    print(f" Battery SOC {results[idx]["log"]["bat_soc"].iloc[-1]}")
    print(f" Supercap SOC {results[idx]["log"]["sc_soc"].iloc[-1]}")
    print(f" Loss fraction {results[idx]["log"]["loss_fraction"].iloc[-1]}")
    print(f" Error fraction {results[idx]["log"]["error_fraction"].iloc[-1]}")

def plot_pareto(
    valid_results,
    front,
    x_col="n_cells",
    y_col="loss_fraction",
    y_scale=100,
    ax=None,
    step=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    ax.scatter(
        valid_results[x_col],
        valid_results[y_col] * y_scale,
        alpha=0.35,
        label="Feasible designs",
    )

    if step:
        ax.step(
            front[x_col],
            front[y_col] * y_scale,
            where="post",
            marker="o",
            linewidth=2,
            label="Pareto front",
        )
    else:
        ax.plot(
            front[x_col],
            front[y_col] * y_scale,
            marker="o",
            linewidth=2,
            label="Pareto front",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()

    return ax