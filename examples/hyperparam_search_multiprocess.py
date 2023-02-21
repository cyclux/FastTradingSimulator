"""_summary_
"""

import concurrent.futures
from frady import TradingEngine


config = {
    "trader": {
        "id": 1,
        "creds_path": "exchange_creds.cfg",
        "use_backend": True,
        "dry_run": False,
        "budget": 1100,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 100,
            "amount_invest_relative": None,
            "buy_limit_strategy": False,
            "window": 180,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": "positive",
            "max_buy_per_asset": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.70,
            "profit_ratio_limit": 1,
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
        "sim_start_delta": None,
        "snapshot_size": 1000,
        "snapshot_amount": 1,
    },
}

hyperparam_search = {
    "config": {
        "study_name": "test10",
        "n_trials": 2,
        "n_jobs": 1,
        "direction": "maximize",
        "storage": "JournalStorage",
        "load_if_exists": True,
        "sampler": "RandomSampler",
        "pruner": "HyperbandPruner",
    },
    "params": {
        "window": {"min": 20, "max": 220, "step": 20},
        "buy_opportunity_factor": {"min": -0.05, "max": 0.25, "step": 0.05},
        "buy_opportunity_boundary": {"min": 0.0, "max": 0.15, "step": 0.05},
        "profit_factor": {"min": 1.05, "max": 2.5, "step": 0.05},
        "amount_invest_fiat": {"min": 50, "max": 250, "step": 50},
        "hold_time_limit": {"min": 1000, "max": 10000, "step": 1000},
        "profit_ratio_limit": {"min": 0.85, "max": 1.1, "step": 0.05},
    },
}

N_WORKERS = 8


def run_wrapper():
    return TradingEngine(config=config).run_sim(hyperparam_search)


if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = [executor.submit(run_wrapper) for worker in range(N_WORKERS)]

        for i in concurrent.futures.as_completed(pool):
            print(f"Return Value: {i.result()}")
