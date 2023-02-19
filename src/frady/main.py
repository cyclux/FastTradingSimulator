"""_summary_
pip install numpy pandas pyarrow pymongo bitfinex-api-py
websockets tensorflow-probability numexpr Bottleneck numba pyyaml
"""
from os import getcwd
import asyncio
from frady import simulator
from frady.simulator import hyperparam_search
from frady.config import Config
from frady.market.backend import Backend
from frady.market.history import MarketHistory
from frady.market.updater import MarketUpdater
from frady.exchange.api import ExchangeAPI
from frady.exchange.websocket import ExchangeWebsocket
from frady.trader import Trader
from frady.utils import connect_api


class TradingEngine:
    """_summary_"""

    def __init__(self, config=None, assets=None):
        print("[INFO] Fast Trading Simulator Beta 0.1.0")
        self.assets_list_symbols = None if assets is None or len(assets) == 0 else assets
        self.config = self.register_config(config)
        self.backend = self.register_backend()
        self.market_history = self.register_market_history()
        self.market_updater_api = self.register_updater()
        self.api = self.connect_api()
        self.exchange_api = self.register_exchange_api()
        self.exchange_ws = self.register_exchange_ws()
        self.trader = self.register_trader()

    def register_config(self, user_config):
        config = Config(user_config)
        return config

    def register_backend(self):
        backend = Backend(fts=self)
        return backend

    def register_updater(self):
        market_updater_api = MarketUpdater(fts=self)
        return market_updater_api

    def register_market_history(self):
        working_dir = getcwd() if self.config.working_dir is None else self.config.working_dir
        market_history = MarketHistory(fts=self, path_current=working_dir)
        return market_history

    def connect_api(self):
        api = {}
        api["bfx_api_priv"] = connect_api(self.config.creds_path, "priv")
        api["bfx_api_pub"] = connect_api(self.config.creds_path, "pub")
        return api

    def register_exchange_api(self):
        exchange_api = ExchangeAPI(fts=self)
        return exchange_api

    def register_exchange_ws(self):
        exchange_ws = ExchangeWebsocket(fts=self)
        return exchange_ws

    def register_trader(self):
        if self.config.run_exchange_api:
            trader = Trader(fts=self)
        else:
            trader = None
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

    async def run_in_jupyter(self):
        # TODO: Transform to sync function. Maybe loop = asyncio.get_event_loop()
        # loop = asyncio.new_event_loop() or loop.run_until_complete(self._run_socket())
        await self.market_history.load_history()
        if self.config.run_exchange_api:
            self.exchange_ws.ws_priv_run()
        if self.config.keep_updated:
            self.run_ws_updater(run_in_jupyter=True)
        return self

    def run_sim(self, optuna_config=None):
        asyncio.run(self.init(is_sim=True))
        if optuna_config is None:
            sim_result = simulator.run(fts=self)
        else:
            sim_result = hyperparam_search.run(self, optuna_config)
        return sim_result
