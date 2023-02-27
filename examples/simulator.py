"""_summary_
"""


import numpy as np

import numba as nb
from tradeforce import TradingEngine

config = {
    "trader": {
        "id": 1,
        "budget": 1100,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 100,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.70,
            "profit_ratio_limit": 1.01,
            "window": 180,
        },
    },
    "market_history": {
        "candle_interval": "5min",
        "history_timeframe": "60days",
        "base_currency": "USD",
        "exchange": "bitfinex",
        "load_history_via": "feather",
        "check_db_consistency": True,
        "dump_to_feather": True,
        "backend": "mongodb",
        "backend_host": "localhost:1234",
        "mongo_collection": "bfx_history_2y",
        "update_history": False,
        "run_exchange_api": True,
        "keep_updated": True,
    },
    "simulation": {
        "snapshot_size": 100000,
        "snapshot_amount": 10,
    },
}

assets = []


@nb.njit()
def buy_strategy(params, window_history_prices_pct):
    buyfactor_row = np.sum(window_history_prices_pct, axis=0)
    buy_opportunity_factor_min = params["buy_opportunity_factor"] - params["buy_opportunity_boundary"]
    buy_opportunity_factor_max = params["buy_opportunity_factor"] + params["buy_opportunity_boundary"]
    buy_options_bool = (buyfactor_row >= buy_opportunity_factor_min) & (buyfactor_row <= buy_opportunity_factor_max)
    if np.any(buy_options_bool):
        buy_option_indices = np.where(buy_options_bool)[0].astype(np.float64)
        buy_option_values = buyfactor_row[buy_options_bool]
        # prefer_performance can be -1, 0 or 1.
        if params["prefer_performance"] == 0:
            buy_option_values = np.absolute(buy_option_values - params["buy_opportunity_factor"])
        buy_option_array = np.vstack((buy_option_indices, buy_option_values))
        buy_option_array = buy_option_array[:, buy_option_array[1, :].argsort()]
        if params["prefer_performance"] == 1:
            # flip axis
            buy_option_array = buy_option_array[:, ::-1]
        buy_option_array_int = buy_option_array[0].astype(np.int64)
    return buy_option_array_int


sim_result = TradingEngine(config=config, assets=assets).run_sim(buy_strategy=buy_strategy)
print(sim_result["profit"])
