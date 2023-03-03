"""_summary_
"""

# from time import perf_counter
import numpy as np
import numba as nb  # type: ignore
import tradeforce.simulator.utils as sim_utils
import tradeforce.simulator.default_strategies as strategies

NB_PARALLEL = False
NB_CACHE = True


@nb.njit(cache=NB_CACHE, parallel=NB_PARALLEL)
def buy_asset(
    buy_option_idx,
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
    amount_invest_crypto_incl_fee, _, amount_fee_buy_fiat = sim_utils.calc_fee(
        amount_invest_crypto, maker_fee, taker_fee, price_current, "buy"
    )
    amount_invest_fiat_incl_fee = amount_invest_fiat - amount_fee_buy_fiat
    budget -= amount_invest_fiat
    bought_asset_params = np.array(
        [
            [
                buy_option_idx,
                1.0,  # placeholder
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


# cache needs to be off in case of a user defined trading strategy function
@nb.njit(cache=False, parallel=NB_PARALLEL, fastmath=True)
def check_buy(params, df_asset_prices_pct, df_asset_performance, buybag, history_prices_row, budget):
    list_buy_options = strategies.buy_strategy(params, df_asset_prices_pct, df_asset_performance)

    for buy_option_idx in list_buy_options:
        price_current = history_prices_row[buy_option_idx]
        if buy_option_idx in buybag[:, 0:1]:

            buybag_idxs = buybag[:, 0:1]
            buybag_idx = np.where(buybag_idxs == buy_option_idx)[0]
            asset_buy_prices = buybag[buybag_idx, 3]
            is_max_buy_per_asset = len(asset_buy_prices) >= params["max_buy_per_asset"]
            if is_max_buy_per_asset:
                continue

        investment_fiat_incl_fee = buybag[:, 5:6]
        fee_fiat = buybag[:, 7:8]
        investment_total = np.sum(investment_fiat_incl_fee + fee_fiat)
        if investment_total >= params["investment_cap"] > 0:
            continue

        if budget >= params["amount_invest_fiat"]:
            bought_asset_params, budget = buy_asset(
                buy_option_idx,
                params["row_idx"],
                price_current,
                params["profit_factor"],
                params["amount_invest_fiat"],
                params["maker_fee"],
                params["taker_fee"],
                budget,
            )
            buybag = np.append(buybag, bought_asset_params, axis=0)
    return buybag
