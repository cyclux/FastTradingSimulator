"""_summary_
"""

import numpy as np
import numba as nb
from tradeforce.simulator.utils import calc_fee

NB_PARALLEL = False
NB_CACHE = False


@nb.njit()
def get_buy_options(buy_options_bool, buyfactor_row, buy_opportunity_factor, prefer_performance):
    # prefer_performance can be -1, 1, and 0.
    buy_option_indices = np.where(buy_options_bool)[0].astype(np.float64)
    buy_option_values = buyfactor_row[buy_options_bool]
    # buy_opportunity_factor == 999 means it is not set
    if prefer_performance == 0 and buy_opportunity_factor != 999:
        buy_option_values = np.absolute(buy_option_values - buy_opportunity_factor)
    buy_option_array = np.vstack((buy_option_indices, buy_option_values))
    buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]
    if prefer_performance == 1:
        # flip axis
        buy_option_array = buy_option_array[:, ::-1]
    buy_option_array_int = buy_option_array[0].astype(np.int64)
    return buy_option_array_int


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def buy_asset(
    buy_option_idx,
    buyfactor_row,
    row_idx,
    price_current,
    profit_factor,
    amount_invest_fiat,
    maker_fee,
    taker_fee,
    budget,
):
    price_profit = price_current * profit_factor
    amount_invest_crypto = amount_invest_fiat / price_current
    amount_invest_crypto_incl_fee, _, amount_fee_buy_fiat = calc_fee(
        amount_invest_crypto, maker_fee, taker_fee, price_current, "buy"
    )
    amount_invest_fiat_incl_fee = amount_invest_fiat - amount_fee_buy_fiat
    budget -= amount_invest_fiat
    bought_asset_params = np.array(
        [
            [
                buy_option_idx,
                buyfactor_row[buy_option_idx],
                row_idx,
                price_current,
                price_profit,
                amount_invest_fiat_incl_fee,
                amount_invest_crypto_incl_fee,
                amount_fee_buy_fiat,
                budget,
            ]
        ]
    )
    return bought_asset_params, budget


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def check_buy(
    list_buy_options,
    buybag,
    investment_cap,
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
):
    # amount_buy_orders = buybag.shape[0]
    # print("before", amount_buy_orders)
    for buy_option_idx in list_buy_options:
        price_current = df_history_prices[row_idx, buy_option_idx]
        if buy_option_idx in buybag[:, 0:1]:

            buybag_idxs = buybag[:, 0:1]
            buybag_idx = np.where(buybag_idxs == buy_option_idx)[0]
            asset_buy_prices = buybag[buybag_idx, 3]
            min_price_bought = np.min(asset_buy_prices)

            is_max_buy_per_asset = len(asset_buy_prices) >= max_buy_per_asset

            is_buy_opportunity = min_price_bought * (1 + buy_opportunity_factor_max) >= price_current
            if is_max_buy_per_asset or not is_buy_opportunity:
                continue

        investment_fiat_incl_fee = buybag[:, 5:6]
        fee_fiat = buybag[:, 7:8]
        investment_total = np.sum(investment_fiat_incl_fee + fee_fiat)
        if investment_total >= investment_cap > 0:
            continue

        if budget >= amount_invest_fiat:
            bought_asset_params, budget = buy_asset(
                buy_option_idx,
                buyfactor_row,
                row_idx,
                price_current,
                profit_factor,
                amount_invest_fiat,
                maker_fee,
                taker_fee,
                budget,
            )
            buybag = np.append(buybag, bought_asset_params, axis=0)
    # amount_buy_orders = buybag.shape[0]
    return buybag
