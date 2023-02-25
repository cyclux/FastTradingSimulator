"""_summary_
"""

import numpy as np
import numba as nb
import numba.typed as nb_types
from tradeforce.simulator.utils import calc_fee, array_diff

NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def sell_asset(
    current_iter,
    current_idx,
    row_idx,
    bought_asset_params,
    soldbag,
    price_current,
    amount_invest_fiat,
    maker_fee,
    taker_fee,
    budget,
):
    amount_invest_crypto = bought_asset_params[6]
    amount_sold_crypto_incl_fee, _, amount_fee_sell_fiat = calc_fee(
        amount_invest_crypto, maker_fee, taker_fee, price_current, "sell"
    )
    amount_sold_fiat_incl_fee = np.round(amount_sold_crypto_incl_fee * price_current, 3)
    amount_profit_fiat = amount_sold_fiat_incl_fee - amount_invest_fiat
    bought_asset_params[8] = budget + amount_sold_fiat_incl_fee
    placeholder_value_crypto_in_fiat = 0.0
    placeholder_total_value = 0.0
    placeholder_amount_buy_orders = 0.0
    sold_asset_params = np.array(
        [
            row_idx,
            price_current,
            amount_sold_fiat_incl_fee,
            amount_sold_crypto_incl_fee,
            amount_fee_sell_fiat,
            amount_profit_fiat,
            placeholder_value_crypto_in_fiat,
            placeholder_total_value,
            placeholder_amount_buy_orders,
            current_iter,
            current_idx,
        ]
    )
    bought_asset_params = np.append(bought_asset_params, sold_asset_params)
    bought_asset_params = np.expand_dims(bought_asset_params, axis=0)

    soldbag = np.append(soldbag, bought_asset_params, axis=0)
    return soldbag


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def check_sell(params, current_iter, current_idx, buybag, soldbag, row_idx, history_prices_row, budget):
    if buybag.shape[0] < 1:
        return soldbag, buybag

    del_buybag_items_list = nb_types.List()
    amount_buy_orders = buybag.shape[0]
    for buybag_row_idx, bought_asset_params in enumerate(buybag):
        buy_option_idx = bought_asset_params[0]
        buy_option_idx_int = np.int64(buy_option_idx)
        row_idx_bought = bought_asset_params[2]
        price_current = history_prices_row[buy_option_idx_int]
        price_bought = bought_asset_params[3]
        price_profit = bought_asset_params[4]
        time_since_buy = row_idx - row_idx_bought
        current_profit_ratio = price_current / price_bought

        ok_to_sell = time_since_buy > params["hold_time_limit"] and current_profit_ratio >= params["profit_ratio_limit"]
        if (price_current >= price_profit) or ok_to_sell:
            # check plausibility and prevent false logic
            # profit gets a max plausible threshold
            if price_current / price_profit > 1.2:
                price_current = price_profit
            soldbag = sell_asset(
                current_iter,
                current_idx,
                row_idx,
                bought_asset_params,
                soldbag,
                price_current,
                params["amount_invest_fiat"],
                params["maker_fee"],
                params["taker_fee"],
                budget,
            )
            budget = soldbag[-1, 8]
            del_buybag_items_list.append(buybag_row_idx)

    if len(del_buybag_items_list) > 0:
        del_buybag_items_array = np.array([x for x in del_buybag_items_list])
        item_row_idx_range = np.array([x for x in range(amount_buy_orders)])
        excluded_del_rows = array_diff(item_row_idx_range, del_buybag_items_array)
        buybag = buybag[excluded_del_rows, :]

    for sold_asset_idx in range(len(del_buybag_items_list)):
        # Calculate buybag crypto value in fiat
        crypto_invested = buybag[:, 6:7].flatten()
        asset_idx = buybag[:, 0:1].flatten().astype(np.int64)
        current_price_buybag = history_prices_row[asset_idx].flatten()
        value_crypto_in_fiat = np.sum(current_price_buybag * crypto_invested)
        sold_asset_reverse_idx = (sold_asset_idx * -1) - 1
        soldbag[sold_asset_reverse_idx:, 15:16] = np.round(value_crypto_in_fiat, 2)
        soldbag[sold_asset_reverse_idx:, 16:17] = np.round(value_crypto_in_fiat + budget, 2)
        soldbag[sold_asset_reverse_idx:, 17:18] = buybag.shape[0]

    return soldbag, buybag
