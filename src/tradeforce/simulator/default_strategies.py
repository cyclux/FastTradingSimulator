import numpy as np
import numba as nb


@nb.njit(cache=True, parallel=False)
def buy_strategy(params, df_asset_prices_pct, df_asset_performance):

    row_idx = np.int64(params["row_idx"] - params["window"])  # init row_idx == 0
    buyfactor_row = df_asset_performance[row_idx]
    # window_history_prices_pct = get_current_window(params, df_asset_prices_pct)
    # buyfactor_row = np.sum(window_history_prices_pct, axis=0)

    buy_opportunity_factor_min = params["buy_opportunity_factor"] - params["buy_opportunity_boundary"]
    buy_opportunity_factor_max = params["buy_opportunity_factor"] + params["buy_opportunity_boundary"]
    buy_options_bool = (buyfactor_row >= buy_opportunity_factor_min) & (buyfactor_row <= buy_opportunity_factor_max)
    if np.any(buy_options_bool):
        buy_option_indices = np.where(buy_options_bool)[0].astype(np.float64)
        buy_option_values = buyfactor_row[buy_options_bool]
        # prefer_performance can be -1, 1, and 0.
        if params["prefer_performance"] == 0:
            buy_option_values = np.absolute(buy_option_values - params["buy_opportunity_factor"])
        buy_option_array = np.vstack((buy_option_indices, buy_option_values))
        buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]
        if params["prefer_performance"] == 1:
            # flip axis
            buy_option_array = buy_option_array[:, ::-1]
        buy_option_array_int = buy_option_array[0].astype(np.int64)
    return buy_option_array_int


@nb.njit(cache=True, parallel=False)
def sell_strategy(params, buybag, history_prices_row):
    buy_option_idxs = buybag[:, 0:1].T.flatten().astype(np.int64)
    prices_current = history_prices_row[buy_option_idxs].reshape((1, -1)).T
    prices_profit = buybag[:, 4:5]
    # check plausibility and prevent false logic
    # profit gets a max plausible threshold
    sanity_check_mask = (prices_current / prices_profit > 1.2).flatten()
    prices_current[sanity_check_mask] = prices_profit[sanity_check_mask]
    times_since_buy = params["row_idx"] - buybag[:, 2:3]
    current_profit_ratios = prices_current / buybag[:, 3:4]
    sell_prices_reached = prices_current >= prices_profit
    ok_to_sells = (times_since_buy > params["hold_time_limit"]) & (
        current_profit_ratios >= params["profit_ratio_limit"]
    )
    sell_assets = (sell_prices_reached | ok_to_sells).flatten()
    return sell_assets, prices_current
