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


@nb.njit(cache=NB_CACHE, parallel=False)
def iter_market_history(
    window,
    df_history_prices_pct,
    snapshot_bounds,
    df_history_prices,
    amount_invest_fiat,
    budget,
    buy_opportunity_factor,
    buy_opportunity_factor_max,
    buy_opportunity_boundary,
    prefer_performance,
    profit_factor,
    investment_cap,
    max_buy_per_asset,
    hold_time_limit,
    profit_ratio_limit,
    maker_fee,
    taker_fee,
    current_idx,
    current_iter,
):
    buybag = np.empty((0, 9), type_float)
    soldbag = np.empty((0, 20), type_float)
    start_idx = snapshot_bounds[0]
    end_idx = snapshot_bounds[1]
    for row_idx, history_prices_row in enumerate(df_history_prices[start_idx:end_idx]):
        # row_idx needs to get shifted by start_idx (start idx of the current snapshot iteration)
        row_idx += start_idx
        soldbag, buybag = check_sell(
            current_iter,
            current_idx,
            buybag,
            soldbag,
            row_idx,
            history_prices_row,
            amount_invest_fiat,
            maker_fee,
            taker_fee,
            budget,
            hold_time_limit,
            profit_ratio_limit,
        )
        budget = set_budget_from_bag(row_idx, budget, soldbag, "soldbag")

        list_buy_options, buyfactor_row = get_buy_options(
            window,
            row_idx,
            df_history_prices_pct,
            buy_opportunity_factor,
            buy_opportunity_boundary,
            prefer_performance,
        )

        buybag = check_buy(
            list_buy_options,
            buybag,
            investment_cap,
            max_buy_per_asset,
            buyfactor_row,
            row_idx,
            history_prices_row,
            buy_opportunity_factor_max,
            profit_factor,
            amount_invest_fiat,
            maker_fee,
            taker_fee,
            budget,
        )
        budget = set_budget_from_bag(row_idx, budget, buybag, "buybag")
        current_idx += 300000

    return soldbag, buybag


@nb.njit(cache=NB_CACHE, parallel=False)
def simulate_trading(sim_params_numba, df_history_prices):
    current_idx = sim_params_numba["index_start"]
    window = sim_params_numba["window"]
    buy_opportunity_factor_max = sim_params_numba["buy_opportunity_factor_max"]
    buy_opportunity_factor = sim_params_numba["buy_opportunity_factor"]
    buy_opportunity_boundary = sim_params_numba["buy_opportunity_boundary"]
    prefer_performance = sim_params_numba["prefer_performance"]
    profit_factor = sim_params_numba["profit_factor"]
    budget = sim_params_numba["budget"]
    amount_invest_fiat = sim_params_numba["amount_invest_fiat"]
    maker_fee = sim_params_numba["maker_fee"]
    taker_fee = sim_params_numba["taker_fee"]
    investment_cap = sim_params_numba["investment_cap"]
    hold_time_limit = sim_params_numba["hold_time_limit"]
    profit_ratio_limit = sim_params_numba["profit_ratio_limit"]
    max_buy_per_asset = sim_params_numba["max_buy_per_asset"]
    snapshot_size = sim_params_numba["snapshot_size"]
    snapshot_amount = np.int64(sim_params_numba["snapshot_amount"])

    if buy_opportunity_factor != 999:
        buy_opportunity_factor_max = buy_opportunity_factor + buy_opportunity_boundary

    # Fill NaN probably not needed as it is done by the data fetch api
    # fill_nan(df_buy_factors)
    # current_idx += window * 300000

    df_history_prices_pct = (df_history_prices[1:, :] - df_history_prices[:-1, :]) / df_history_prices[1:, :]
    df_history_prices = df_history_prices[1:]

    snapshot_idx_boundary = df_history_prices_pct.shape[0]
    snapshot_size, snapshot_amount = sanitize_snapshot_params(snapshot_size, snapshot_amount, snapshot_idx_boundary)
    snapshot_start_idxs = get_snapshot_indices(window, snapshot_idx_boundary, snapshot_amount, snapshot_size)
    profit_snapshot_list = np.empty(snapshot_amount, type_float)
    soldbag_all_snapshots = np.empty((0, 20), type_float)
    # index_start = np.float64(history_buy_factors.index[0])
    for current_iter, snapshot_idx in enumerate(snapshot_start_idxs, 1):
        snapshot_bounds = (snapshot_idx, snapshot_idx + snapshot_size)
        soldbag, buybag = iter_market_history(
            window,
            df_history_prices_pct,
            snapshot_bounds,
            df_history_prices,
            amount_invest_fiat,
            budget,
            buy_opportunity_factor,
            buy_opportunity_factor_max,
            buy_opportunity_boundary,
            prefer_performance,
            profit_factor,
            investment_cap,
            max_buy_per_asset,
            hold_time_limit,
            profit_ratio_limit,
            maker_fee,
            taker_fee,
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
