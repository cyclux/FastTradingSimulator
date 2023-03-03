"""_summary_
"""

from tradeforce import TradingEngine

config = {
    "trader": {
        "id": 3,
        # "creds_path": "exchange_creds.cfg",
        "budget": 0,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
            "amount_invest_fiat": 1000,
            "investment_cap": 0,
            "buy_opportunity_factor": 0.10,
            "buy_opportunity_boundary": 0.05,
            "prefer_performance": 1,
            "hold_time_limit": 1000,
            "profit_factor": 1.10,
            "profit_ratio_limit": 1.01,
            "window": 160,
        },
    },
    "market_history": {
        "candle_interval": "5min",
        "history_timeframe": "720days",
        "base_currency": "USD",
        "exchange": "bitfinex",
        "load_history_via": "API",
        "check_db_consistency": True,
        "dump_to_feather": True,
        "backend": "mongodb",
        "backend_host": "localhost:1234",
        "mongo_collection": "bfx_history_2y",
        "update_history": False,
        "run_exchange_api": False,
        "keep_updated": True,
    },
}

# To provide a subset, TradingEngine can also receive the optional argument "assets".
# This has to be a list of asset symbols. For example: assets = ["BTC", "ETH", "XRP"]
TradingEngine(config).run()
