import yaml


def load_yaml_config():
    with open("config.yaml", "r", encoding="utf8") as stream:
        yaml_config = yaml.safe_load(stream)
    return yaml_config


config_input = load_yaml_config()
# print(yaml_return)


# update_history = pd.json_normalize(config_input).to_dict("records")
# print(update_history[0])


def flatten_dict(input_dict):
    output_dict = {}

    def flatten(input_dict, key=""):
        if isinstance(input_dict, dict):
            for key in input_dict:
                flatten(input_dict[key], key)
        else:
            output_dict[key] = input_dict

    flatten(input_dict)
    return output_dict


config = {
    "trader": {
        "id": 1,
        "creds_path": "exchange_creds.cfg",
        "use_backend": True,
        "dry_run": False,
        "budget": 0,
        "maker_fee": 0.10,
        "taker_fee": 0.20,
        "strategy": {
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
        "snapshot_size": -1,
        "snapshot_amount": 1,
    },
}

config = flatten_dict(config)

update_history = config.get("run_exchange_api", False)
print(update_history)
