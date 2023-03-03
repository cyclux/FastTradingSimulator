"""_summary_
"""

import numpy as np
import numba as nb

from tradeforce.simulator.utils import calc_fee
import tradeforce.simulator.default_strategies as strategies

NB_PARALLEL = False
NB_CACHE = True

type_float = nb.typeof(0.1)


@nb.njit(cache=NB_CACHE, parallel=False)
def sell_asset(
    params,
    assets_to_sell,
    soldbag,
    prices_current,
    budget,
):

    amount_assets_sell = assets_to_sell.shape[0]
    sold_assets = np.empty((amount_assets_sell, 15), type_float)
    amounts_invest_crypto = assets_to_sell[:, 6:7]
    amount_sold_crypto_incl_fee, _, amount_fee_sell_fiat = calc_fee(
        amounts_invest_crypto, params["maker_fee"], params["taker_fee"], prices_current, "sell"
    )
    amount_sold_fiat_incl_fee = amount_sold_crypto_incl_fee * prices_current
    amount_profit_fiat = amount_sold_fiat_incl_fee - params["amount_invest_fiat"]

    # update budget with new profit
    new_budget = np.sum(amount_sold_fiat_incl_fee) + budget
    new_budget_reshaped = np.broadcast_to(new_budget, (amount_assets_sell, 1))
    assets_to_sell[:, 8:9] = new_budget_reshaped
    row_idx_sold = np.broadcast_to(params["row_idx"], (amount_assets_sell, 1))
    sold_assets[:, 0:9] = assets_to_sell
    sold_assets[:, 9:10] = row_idx_sold
    sold_assets[:, 10:11] = prices_current
    sold_assets[:, 11:12] = amount_sold_fiat_incl_fee
    sold_assets[:, 12:13] = amount_sold_crypto_incl_fee
    sold_assets[:, 13:14] = amount_fee_sell_fiat
    sold_assets[:, 14:15] = amount_profit_fiat

    soldbag = np.append(soldbag, sold_assets, axis=0)
    return soldbag


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=False, fastmath=True)
def check_sell(params, buybag, soldbag, history_prices_row, budget):
    amount_buy_orders = buybag.shape[0]
    if amount_buy_orders < 1:
        return soldbag, buybag

    sell_assets, prices_current = strategies.sell_strategy(params, buybag, history_prices_row)

    assets_to_sell = buybag[sell_assets, :]
    if assets_to_sell.shape[0] > 0:
        soldbag = sell_asset(
            params,
            assets_to_sell,
            soldbag,
            prices_current[sell_assets, :],
            budget,
        )
        buybag = buybag[~sell_assets, :]
    return soldbag, buybag
