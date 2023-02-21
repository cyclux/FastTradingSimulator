"""_summary_
pip install numpy pandas pyarrow pymongo bitfinex-api-py
websockets tensorflow-probability numexpr Bottleneck numba pyyaml
"""

import os
import logging
import asyncio
from tradeforce import simulator
from tradeforce.simulator import hyperparam_search
from tradeforce.config import Config
from tradeforce.market.backend import Backend
from tradeforce.market.history import MarketHistory
from tradeforce.market.updater import MarketUpdater
from tradeforce.exchange.api import ExchangeAPI
from tradeforce.exchange.websocket import ExchangeWebsocket
from tradeforce.trader import Trader
from tradeforce.utils import connect_api


class TradingEngine:
    """_summary_"""

    def __init__(self, config=None, assets=None):
        self.logging = logging
        self.log = self.config_logger()
        self.log.info("Fast Trading Simulator Beta 0.1.0")
        self.assets_list_symbols = None if assets is None or len(assets) == 0 else assets
        self.config = self.register_config(config)
        self.trader = self.register_trader()
        self.backend = self.register_backend()
        self.market_history = self.register_market_history()
        self.market_updater_api = self.register_updater()
        self.api = self.connect_api()
        self.exchange_api = self.register_exchange_api()
        self.exchange_ws = self.register_exchange_ws()

    def config_logger(self):
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
        log = logging.getLogger(__name__)
        return log

    def register_config(self, user_config):
        config = Config(root=self, config_input=user_config)
        return config

    def register_backend(self):
        backend = Backend(root=self)
        return backend

    def register_updater(self):
        market_updater_api = MarketUpdater(root=self)
        return market_updater_api

    def register_market_history(self):
        working_dir = os.getcwd() if self.config.working_dir is None else self.config.working_dir
        market_history = MarketHistory(root=self, path_current=working_dir)
        return market_history

    def connect_api(self):
        api = {}
        api["bfx_api_priv"] = connect_api(self.config.creds_path, "priv")
        api["bfx_api_pub"] = connect_api(self.config.creds_path, "pub")
        return api

    def register_exchange_api(self):
        exchange_api = ExchangeAPI(root=self)
        return exchange_api

    def register_exchange_ws(self):
        exchange_ws = ExchangeWebsocket(root=self)
        return exchange_ws

    def register_trader(self):
        trader = Trader(root=self)
        return trader

    async def init(self, is_sim=False):
        await self.market_history.load_history()
        if self.config.run_exchange_api and not is_sim:
            self.exchange_ws.ws_priv_run()
        return self

    def run_ws_updater(self, run_in_jupyter=False):
        if not run_in_jupyter:
            loop_ws_updater = asyncio.new_event_loop()
            asyncio.set_event_loop(loop_ws_updater)
        self.exchange_ws.ws_run()

    def run(self):
        asyncio.run(self.init())
        if self.config.keep_updated:
            self.run_ws_updater()
        return self

    def run_sim(self, optuna_config=None):
        asyncio.run(self.init(is_sim=True))
        if optuna_config is None:
            sim_result = simulator.run(self)
        else:
            sim_result = hyperparam_search.run(self, optuna_config)
        return sim_result

    async def run_jupyter(self):
        await self.market_history.load_history()
        if self.config.run_exchange_api:
            self.exchange_ws.ws_priv_run()
        if self.config.keep_updated:
            self.run_ws_updater(run_in_jupyter=True)
        return self

    async def run_sim_jupyter(self, optuna_config=None):
        await self.market_history.load_history()
        if optuna_config is None:
            sim_result = simulator.run(self)
        else:
            sim_result = hyperparam_search.run(self, optuna_config)
        return sim_result
