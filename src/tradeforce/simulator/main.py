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
# 15 value_crypto_in_fiat
# 16 total value of account
# 17 amount buy orders
# 18 current snapshot iteration
# 19 current_idx

Returns:
    _type_: _description_
"""

import numpy as np
import numba as nb

from tradeforce.utils import get_timedelta
from tradeforce.simulator.utils import get_snapshot_indices, calc_metrics, to_numba_dict, sanitize_snapshot_params
from tradeforce.simulator.buys import check_buy, get_buy_options
from tradeforce.simulator.sells import check_sell


# FIXME: Add money which is in assets on the market

# type_int = nb.typeof(1)
type_float = nb.typeof(0.1)
# type_array_1d_float = nb.typeof(np.array([0.1]))
# type_array_2d_float = nb.typeof(np.array([[0.1]]))

NB_PARALLEL = False
NB_CACHE = False


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


@nb.njit(cache=NB_CACHE, parallel=False)
def iter_market_history(
    params,
    df_history_prices_pct,
    snapshot_bounds,
    df_history_prices,
    current_idx,
    current_iter,
):
    budget = params["budget"]
    buybag = np.empty((0, 9), type_float)
    soldbag = np.empty((0, 20), type_float)
    start_idx = snapshot_bounds[0]
    end_idx = snapshot_bounds[1]
    for row_idx, history_prices_row in enumerate(df_history_prices[start_idx:end_idx]):
        # row_idx needs to get shifted by start_idx (start idx of the current snapshot iteration)
        row_idx += start_idx
        soldbag, buybag = check_sell(
            params, current_iter, current_idx, buybag, soldbag, row_idx, history_prices_row, budget
        )
        budget = set_budget_from_bag(row_idx, budget, soldbag, "soldbag")

        list_buy_options = get_buy_options(params, row_idx, df_history_prices_pct)

        buybag = check_buy(
            params,
            list_buy_options,
            buybag,
            row_idx,
            history_prices_row,
            budget,
        )
        budget = set_budget_from_bag(row_idx, budget, buybag, "buybag")
        current_idx += 300000

    return soldbag, buybag


@nb.njit(cache=NB_CACHE, parallel=False)
def simulate_trading(params, df_history_prices):
    current_idx = params["index_start"]
    # Fill NaN probably not needed as it is done by the data fetch api
    # fill_nan(df_buy_factors)
    # current_idx += window * 300000

    df_history_prices_pct = (df_history_prices[1:, :] - df_history_prices[:-1, :]) / df_history_prices[1:, :]
    df_history_prices = df_history_prices[1:]

    snapshot_idx_boundary = df_history_prices_pct.shape[0]
    snapshot_size, snapshot_amount = sanitize_snapshot_params(params, snapshot_idx_boundary)
    snapshot_start_idxs = get_snapshot_indices(params["window"], snapshot_idx_boundary, snapshot_amount, snapshot_size)
    profit_snapshot_list = np.empty(snapshot_amount, type_float)
    soldbag_all_snapshots = np.empty((0, 20), type_float)
    # index_start = np.float64(history_buy_factors.index[0])
    for current_iter, snapshot_idx in enumerate(snapshot_start_idxs, 1):
        snapshot_bounds = (snapshot_idx, snapshot_idx + snapshot_size)
        # params["index_start"] = current_idx + (snapshot_idx * 300000)

        soldbag, buybag = iter_market_history(
            params,
            df_history_prices_pct,
            snapshot_bounds,
            df_history_prices,
            current_idx + (snapshot_idx * 300000),
            current_iter,
        )
        profit_snapshot_list[current_iter - 1] = calc_metrics(soldbag)
        soldbag_all_snapshots = np.vstack((soldbag_all_snapshots, soldbag))

    profit_total_std = np.std(profit_snapshot_list)
    profit_total = np.int64(np.mean(profit_snapshot_list) - profit_total_std)

    return profit_total, soldbag_all_snapshots, buybag


def print_sim_details(root, bfx_history):
    history_begin = bfx_history.index[0]
    history_end = bfx_history.index[-1]
    history_delta = get_timedelta(history_end - history_begin, unit="ms")["datetime"]
    root.log.info(
        "Starting simulation beginning from %s to %s | Timeframe: %s", history_begin, history_end, history_delta
    )


def prepare_sim(root):
    # TODO: provide start and timeframe for simulation
    # window = int(root.config.window)
    sim_start_delta = root.config.sim_start_delta
    bfx_history = root.market_history.get_market_history(start=sim_start_delta, metrics=["o"], fill_na=True)
    # bfx_history_pct = root.market_history.get_market_history(
    #     start=sim_start_delta, metrics=["o"], fill_na=True, pct_change=True, pct_as_factor=False
    # )
    # history_buy_factors = bfx_history_pct.rolling(window=window, step=1, min_periods=1).sum(
    #     engine="numba", engine_kwargs={"parallel": True, "cache": True}
    # )
    print_sim_details(root, bfx_history)
    return bfx_history.to_numpy()


def run(root, bfx_history=None, sim_config=None):
    if bfx_history is None:
        bfx_history = prepare_sim(root)
    if sim_config is None:
        sim_config = to_numba_dict(root.config.to_dict())

    total_profit, trades_history, buy_log = simulate_trading(sim_config, bfx_history)
    sim_result = {"profit": total_profit, "trades": trades_history, "buy_log": buy_log}
    return sim_result
