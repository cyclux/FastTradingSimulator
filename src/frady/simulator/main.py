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

from frady.utils import get_timedelta
from frady.simulator.utils import get_snapshot_indices, calc_metrics, to_numba_dict
from frady.simulator.buys import check_buy, get_buy_options
from frady.simulator.sells import check_sell


# FIXME: Add money which is in assets on the market

# type_int = nb.typeof(1)
type_float = nb.typeof(0.1)
# type_array_1d_float = nb.typeof(np.array([0.1]))
# type_array_2d_float = nb.typeof(np.array([[0.1]]))

NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=False)
def iter_market_history(
    df_buy_factors,
    snapshot_bounds,
    df_history_prices,
    amount_invest_fiat,
    budget,
    buy_opportunity_factor,
    buy_opportunity_factor_max,
    buy_opportunity_factor_min,
    prefer_performance,
    profit_factor,
    asset_buy_limit,
    buy_limit_strategy,
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
    for row_idx, buyfactor_row in enumerate(df_buy_factors[start_idx:end_idx]):
        soldbag, buybag = check_sell(
            current_iter,
            current_idx,
            buybag,
            soldbag,
            row_idx,
            df_history_prices,
            amount_invest_fiat,
            maker_fee,
            taker_fee,
            budget,
            asset_buy_limit,
            buy_limit_strategy,
            hold_time_limit,
            profit_ratio_limit,
        )

        if soldbag.shape[0] > 0:
            row_idx_last_sell = soldbag[-1, 9]
            if row_idx_last_sell == row_idx:
                budget = soldbag[-1, 8]

        buy_options_bool = (buyfactor_row >= buy_opportunity_factor_min) & (buyfactor_row <= buy_opportunity_factor_max)
        if np.any(buy_options_bool):
            list_buy_options = get_buy_options(
                buy_options_bool, buyfactor_row, buy_opportunity_factor, prefer_performance
            )
            buybag = check_buy(
                list_buy_options,
                buybag,
                asset_buy_limit,
                buy_limit_strategy,
                max_buy_per_asset,
                buyfactor_row,
                row_idx,
                df_history_prices,
                buy_opportunity_factor_max,
                profit_factor,
                amount_invest_fiat,
                maker_fee,
                taker_fee,
                budget,
            )
            if buybag.shape[0] > 0:
                row_idx_last_buy = buybag[-1, 2]
                if row_idx_last_buy == row_idx:
                    budget = buybag[-1, 8]

        current_idx += 300000

    return soldbag, buybag


@nb.njit(cache=NB_CACHE, parallel=False)
def simulate_trading(sim_params_numba, df_buy_factors, df_history_prices):
    current_idx = sim_params_numba["index_start"]
    window = sim_params_numba["window"]
    buy_opportunity_factor_max = sim_params_numba["buy_opportunity_factor_max"]
    buy_opportunity_factor_min = sim_params_numba["buy_opportunity_factor_min"]
    buy_opportunity_factor = sim_params_numba["buy_opportunity_factor"]
    buy_opportunity_boundary = sim_params_numba["buy_opportunity_boundary"]
    prefer_performance = sim_params_numba["prefer_performance"]
    profit_factor = sim_params_numba["profit_factor"]
    budget = sim_params_numba["budget"]
    amount_invest_fiat = sim_params_numba["amount_invest_fiat"]
    maker_fee = sim_params_numba["maker_fee"]
    taker_fee = sim_params_numba["taker_fee"]
    buy_limit_strategy = sim_params_numba["buy_limit_strategy"]
    hold_time_limit = sim_params_numba["hold_time_limit"]
    profit_ratio_limit = sim_params_numba["profit_ratio_limit"]
    max_buy_per_asset = sim_params_numba["max_buy_per_asset"]
    snapshot_size = sim_params_numba["snapshot_size"]
    snapshot_amount = np.int64(sim_params_numba["snapshot_amount"])

    asset_buy_limit = budget // amount_invest_fiat
    if buy_opportunity_factor != 999:
        buy_opportunity_factor_min = buy_opportunity_factor - buy_opportunity_boundary
        buy_opportunity_factor_max = buy_opportunity_factor + buy_opportunity_boundary

    # Fill NaN probably not needed as it is done by the data fetch api
    # fill_nan(df_buy_factors)

    df_buy_factors = df_buy_factors[window:]
    df_history_prices = df_history_prices[window:]
    # current_idx += window * 300000

    snapshot_idx_boundary = df_buy_factors.shape[0]
    if snapshot_size <= 0:
        snapshot_size = -1
    if snapshot_amount <= 0:
        snapshot_amount = 1
    if snapshot_amount == 1 and snapshot_size == -1:
        snapshot_size = snapshot_idx_boundary
    if snapshot_amount > 1 and snapshot_size == -1:
        snapshot_size = snapshot_idx_boundary // snapshot_amount
    snapshot_bounds = get_snapshot_indices(snapshot_idx_boundary, snapshot_amount, snapshot_size)

    profit_snapshot_list = np.empty(snapshot_amount, type_float)
    soldbag_all_snapshots = np.empty((0, 20), type_float)

    # index_start = np.float64(history_buy_factors.index[0])
    for current_iter, snapshot_idx in enumerate(snapshot_bounds, 1):
        snapshot_bounds = (snapshot_idx, snapshot_idx + snapshot_size)
        soldbag, buybag = iter_market_history(
            df_buy_factors,
            snapshot_bounds,
            df_history_prices,
            amount_invest_fiat,
            budget,
            buy_opportunity_factor,
            buy_opportunity_factor_max,
            buy_opportunity_factor_min,
            prefer_performance,
            profit_factor,
            asset_buy_limit,
            buy_limit_strategy,
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


def run(fts):
    # TODO: provide start and timeframe for simulation
    sim_start_delta = fts.config.sim_start_delta
    # sim_timeframe = get_timedelta(fts.config.sim_timeframe)
    bfx_history = fts.market_history.get_market_history(start=sim_start_delta, metrics=["o"], fill_na=True)
    bfx_history_pct = fts.market_history.get_market_history(
        start=sim_start_delta, metrics=["o"], fill_na=True, pct_change=True, pct_as_factor=False
    )
    history_begin = bfx_history.index[0]
    history_end = bfx_history.index[-1]
    history_delta = get_timedelta(history_end - history_begin, unit="ms")["datetime"]

    print(f"[INFO] Starting simulation beginning from {history_begin} to {history_end} | Timeframe: {history_delta}")
    window = int(fts.config.window)
    history_buy_factors = bfx_history_pct.rolling(window=window, step=1, min_periods=1).sum(
        engine="numba", engine_kwargs={"parallel": True, "cache": True}
    )
    total_profit, trades_history, buy_log = simulate_trading(
        to_numba_dict(fts.config.as_dict()), history_buy_factors.to_numpy(), bfx_history.to_numpy()
    )
    sim_result = {"profit": total_profit, "trades": trades_history, "buy_log": buy_log}
    return sim_result
