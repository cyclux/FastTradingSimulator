"""_summary_
"""

import yaml

SIM_RELEVANT_ENTRIES = [
    "index_start",
    "window",
    "buy_opportunity_factor",
    "buy_opportunity_factor_max",
    "buy_opportunity_factor_min",
    "buy_opportunity_boundary",
    "prefer_performance",
    "profit_factor",
    "budget",
    "amount_invest_fiat",
    "maker_fee",
    "taker_fee",
    "investment_cap",
    "hold_time_limit",
    "profit_ratio_limit",
    "max_buy_per_asset",
    "snapshot_size",
    "snapshot_amount",
]


def load_yaml_config():
    with open("config.yaml", "r", encoding="utf8") as stream:
        yaml_config = yaml.safe_load(stream)
    return yaml_config


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


class Config:
    """_summary_"""

    # TODO: Add sanity check
    def __init__(self, root, config_input):
        self.log = root.logging.getLogger(__name__)
        config_type = "config.yaml" if config_input is None else "dict"
        self.log.info("Loading config via %s", config_type)

        if config_input is None:
            config_input = load_yaml_config()

        config_input = flatten_dict(config_input)

        self.working_dir = config_input.get("working_dir", None)
        self.update_history = config_input.get("update_history", False)
        self.run_exchange_api = config_input.get("run_exchange_api", False)
        self.keep_updated = config_input.get("keep_updated", False)

        self.exchange = config_input.get("exchange", "bitfinex")
        self.load_history_via = config_input.get("load_history_via", "feather").lower()
        self.check_db_consistency = config_input.get("check_db_consistency", False)
        self.dump_to_feather = config_input.get("dump_to_feather", False)
        self.candle_interval = config_input.get("candle_interval", "5min")
        self.base_currency = config_input.get("base_currency", "USD")
        self.history_timeframe = config_input.get("history_timeframe", "120days")
        self.backend = config_input.get("backend", "mongodb").lower()
        self.backend_host = config_input.get("backend_host", "localhost:1234")
        self.backend_user = config_input.get("backend_user", None)
        self.backend_password = config_input.get("backend_password", None)
        self.mongo_exchange_db = config_input.get("mongo_exchange_db", "bitfinex_db")
        self.mongo_collection = config_input.get("mongo_collection", "bfx_history")
        self.creds_path = config_input.get("creds_path", "")
        self.relevant_assets_cap = config_input.get("relevant_assets_cap", 100)

        self.trader_id = config_input.get("id", 1)
        self.window = config_input.get("window", 20) * 60 // 5
        self.budget = float(config_input.get("budget", 0))
        self.buy_opportunity_factor = config_input.get("buy_opportunity_factor", 0.0)
        self.buy_opportunity_boundary = config_input.get("buy_opportunity_boundary", 0.02)
        self.buy_opportunity_factor_min = self.buy_opportunity_factor - self.buy_opportunity_boundary
        self.buy_opportunity_factor_max = self.buy_opportunity_factor + self.buy_opportunity_boundary
        self.profit_factor = config_input.get("profit_factor", 0.05)
        self.profit_ratio_limit = config_input.get("profit_ratio_limit", 1.01)
        self.prefer_performance = config_input.get("prefer_performance", "center")
        self.amount_invest_fiat = config_input.get("amount_invest_fiat", None)
        self.max_buy_per_asset = config_input.get("max_buy_per_asset", 1)
        self.hold_time_limit = config_input.get("hold_time_limit", 20000)
        self.investment_cap = config_input.get("investment_cap", 0)
        self.maker_fee = config_input.get("maker_fee", 0.10)
        self.taker_fee = config_input.get("taker_fee", 0.20)
        self.use_backend = config_input.get("use_backend", True)
        self.is_simulation = config_input.get("dry_run", False)
        # Simulator specific
        self.index_start = config_input.get("index_start", 0)
        self.snapshot_size = config_input.get("snapshot_size", -1)
        self.snapshot_amount = config_input.get("snapshot_amount", 1)
        self.sim_start_delta = config_input.get("sim_start_delta", None)
        self.sim_timeframe = config_input.get("sim_timeframe", None)

        self.assets_excluded = [
            "UDC",
            "UST",
            "EUT",
            "EUS",
            "MIM",
            "PAX",
            "DAI",
            "TSD",
            "TERRAUST",
            "LEO",
            "WBT",
            "RBT",
            "ETHW",
            "SHIB",
            "ETH2X",
            "SPELL",
        ]

    def to_dict(self, for_sim=True):
        attr_to_dict = self.__dict__
        if for_sim:
            attr_to_dict = {key: val for (key, val) in attr_to_dict.items() if key in SIM_RELEVANT_ENTRIES}
        return attr_to_dict
