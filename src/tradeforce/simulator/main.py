"""_summary_
# 0 idx of asset row -> symbol of asset
# 1 buy opportunity factor
# 2 row_idx of buy opportunity
# 3 price buy
# 4 price including profit
# 5 amount of fiat invested, fee included
# 6 amount of crypto invested, fee included
# 7 amount fee buy order in fiat (probably taker fee)
# 8 budget to invest
#
# 9 row_idx of sell opportunity
# 10 price sell
# 11 amount of fiat sold, fee included
# 12 amount of crypto sold, fee included
# 13 amount fee sell order in fiat (probably maker fee)
# 14 amount profit in fiat

Returns:
    _type_: _description_
"""

# from time import perf_counter

import numpy as np
import numba as nb  # type: ignore

from tradeforce.utils import get_timedelta
import tradeforce.simulator.utils as sim_utils
from tradeforce.simulator.buys import check_buy
from tradeforce.simulator.sells import check_sell


# type_int = nb.typeof(1)
type_float = nb.typeof(0.1)
# type_array_1d_float = nb.typeof(np.array([0.1]))
# type_array_2d_float = nb.typeof(np.array([[0.1]]))

NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=False)
def set_budget_from_bag(row_idx, budget, bag, bag_type):
    # bag_row_idx of 2 == row_idx of buy opportunity
    # bag_row_idx of 9 == row_idx of sell opportunity
    bag_row_idx = 2 if bag_type == "buybag" else 9
    if bag.shape[0] > 0:
        row_idx_last_tx = bag[-1, bag_row_idx]
        if row_idx_last_tx == row_idx:
            budget = bag[-1, 8]
    return budget


# cache needs to be off in case of a user defined trading strategy function
# @perf.jit_timer
@nb.njit(cache=False, parallel=False)
def iter_market_history(params, snapshot_bounds, asset_prices, asset_prices_pct, asset_performance):
    budget = params["budget"]
    buybag = np.empty((0, 9), type_float)
    soldbag = np.empty((0, 15), type_float)
    start_idx = snapshot_bounds[0]
    end_idx = snapshot_bounds[1]
    for row_idx, history_prices_row in enumerate(asset_prices[start_idx:end_idx]):
        # row_idx needs to get shifted by start_idx (start idx of the current snapshot iteration)
        row_idx += start_idx
        params["row_idx"] = row_idx

        soldbag, buybag = check_sell(params, buybag, soldbag, history_prices_row, budget)
        budget = set_budget_from_bag(row_idx, budget, soldbag, "soldbag")
        buybag = check_buy(params, asset_prices_pct, asset_performance, buybag, history_prices_row, budget)
        budget = set_budget_from_bag(row_idx, budget, buybag, "buybag")

    return soldbag, buybag


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=False)
def simulate_trading(params, asset_prices, asset_prices_pct, asset_performance):
    # strategy = nb.jit(nopython=True)(strategy)
    # Fill NaN probably not needed as it is done by the data fetch api
    # fill_nan(df_buy_factors)

    snapshot_idx_boundary = asset_prices.shape[0]
    snapshot_size, snapshot_amount = sim_utils.sanitize_snapshot_params(params, snapshot_idx_boundary)
    snapshot_start_idxs = sim_utils.get_snapshot_indices(
        params["window"], snapshot_idx_boundary, snapshot_amount, snapshot_size
    )
    profit_snapshot_list = np.empty(snapshot_amount, type_float)
    soldbag_all_snapshots = np.empty((0, 15), type_float)
    # initialize row_idx within params -> provide access to sub-functions
    params["row_idx"] = 0.0
    for current_iter, snapshot_idx in enumerate(snapshot_start_idxs):
        snapshot_bounds = (snapshot_idx, snapshot_idx + snapshot_size)

        # with nb.objmode(time1="f8"):
        #     time1 = perf_counter()

        soldbag, buybag = iter_market_history(
            params, snapshot_bounds, asset_prices, asset_prices_pct, asset_performance
        )

        # with nb.objmode():
        #     print("time iter_market_history:", perf_counter() - time1)

        profit_snapshot_list[current_iter] = sim_utils.calc_metrics(soldbag)
        soldbag_all_snapshots = np.vstack((soldbag_all_snapshots, soldbag))

    profit_total_std = np.std(profit_snapshot_list)
    profit_total = np.int64(np.mean(profit_snapshot_list) - profit_total_std)

    return profit_total, soldbag_all_snapshots, buybag


def print_sim_details(root, asset_prices):
    history_begin = asset_prices.index[0]
    history_end = asset_prices.index[-1]
    history_delta = get_timedelta(history_end - history_begin, unit="ms")["datetime"]
    root.log.info(
        "Starting simulation beginning from %s to %s | Timeframe: %s", history_begin, history_end, history_delta
    )
    root.log.info(
        "Simulation is split into %s snapshots with size %s each.",
        root.config.snapshot_amount,
        root.config.snapshot_size,
    )


def pre_process_default(config, market_history):
    # TODO: provide start and timeframe for simulation: sim_start_delta ?
    sim_start_delta = config.sim_start_delta
    asset_prices = market_history.get_market_history(start=sim_start_delta, metrics=["o"], fill_na=True)

    asset_prices_pct = market_history.get_market_history(
        start=sim_start_delta, metrics=["o"], fill_na=True, pct_change=True, pct_as_factor=False, pct_first_row=0
    )

    lower_threshold = 0.000005
    upper_threshold = 0.999995
    quantiles = asset_prices_pct.stack().quantile([lower_threshold, upper_threshold])
    asset_prices_pct[asset_prices_pct > quantiles[upper_threshold]] = quantiles[upper_threshold]
    asset_prices_pct[asset_prices_pct < quantiles[lower_threshold]] = quantiles[lower_threshold]

    window = int(config.window)
    asset_market_performance = asset_prices_pct.rolling(window=window, step=1, min_periods=window).sum(
        engine="numba", engine_kwargs={"parallel": True, "cache": True}
    )[window - 1 :]

    preprocess_return = {
        "asset_prices": asset_prices,
        "asset_prices_pct": asset_prices_pct,
        "asset_performance": asset_market_performance,
    }
    return preprocess_return


def run(root, asset_prices=None, sim_config=None, pre_process=None):
    pre_process = pre_process_default if pre_process is None else pre_process
    if asset_prices is None:
        preprocess_result = pre_process(root.config, root.market_history)
    if sim_config is None:
        sim_config = sim_utils.to_numba_dict(root.config.to_dict())

    asset_prices = preprocess_result.get("asset_prices", None)
    asset_prices_pct = preprocess_result.get("asset_prices_pct", None)
    asset_performance = preprocess_result.get("asset_performance", None)

    print_sim_details(root, asset_prices)

    if asset_prices is not None:
        asset_prices = asset_prices.to_numpy()
    if asset_prices_pct is not None:
        asset_prices_pct = asset_prices_pct.to_numpy()
    if asset_performance is not None:
        asset_performance = asset_performance.to_numpy()

    total_profit, trades_history, buy_log = simulate_trading(
        sim_config, asset_prices, asset_prices_pct, asset_performance
    )
    sim_result = {"profit": total_profit, "trades": trades_history, "buy_log": buy_log}
    return sim_result
