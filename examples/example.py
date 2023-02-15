"""_summary_
"""

import fatrasi as fts

config = {
    # trading strategy
    "trader_id": 1,
    "use_backend": True,
    "dry_run": False,
    "amount_invest_fiat": 1000,
    "amount_invest_relative": None,
    "buy_limit_strategy": False,
    "buy_opportunity_factor": 0.10,
    "buy_opportunity_boundary": 0.10,
    "prefer_performance": "positive",
    "max_buy_per_asset": 1,
    "hold_time_limit": 12000,
    "profit_factor": 1.70,
    "profit_ratio_limit": 1,
    "window": 160,
    "exchange_fee": 0.20,
    "budget": 0,
    # market history config
    "asset_interval": "5min",
    "history_timeframe": "60days",
    "base_currency": "USD",
    "exchange": "bitfinex",
    "load_history_via": "feather",
    "check_db_consistency": True,
    "dump_to_feather": True,
    "backend": "mongodb",
    "backend_host": "localhost:1234",
    "mongo_collection": "bfx_history_test",
    "creds_path": "exchange_creds.cfg",
    "update_history": False,
    "run_exchange_api": True,
    "keep_updated": True,
}

assets = []  # ["BTC", "ETH", "XMR"]
fts.TradingEngine(config, assets).run()
