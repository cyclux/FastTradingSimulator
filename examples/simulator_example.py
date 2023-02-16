"""_summary_
"""

from frady import TradingEngine

config = {
    # trading strategy
    "trader_id": 1,
    "creds_path": "exchange_creds.cfg",
    "use_backend": True,
    "dry_run": False,
    "amount_invest_fiat": 100,
    "amount_invest_relative": None,
    "buy_limit_strategy": False,
    "buy_opportunity_factor": 0.10,
    "buy_opportunity_boundary": 0.05,
    "prefer_performance": "positive",
    "max_buy_per_asset": 1,
    "hold_time_limit": 1000,
    "profit_factor": 1.70,
    "profit_ratio_limit": 1,
    "window": 180,
    "exchange_fee": 0.15,
    "budget": 1100,
    # history config
    "candle_interval": "5min",
    "history_timeframe": "750days",
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
    # simulator config
    "snapshot_size": -1,
    "snapshot_amount": 1,
}

assets = []  # ["BTC", "ETH", "XMR"]
sim_result = TradingEngine(config, assets).run_sim()
print(sim_result["profit"])
