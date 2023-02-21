"""_summary_

Returns:
    _type_: _description_
"""

import sys
import numpy as np
import pandas as pd
from tradeforce.utils import convert_symbol_str
from tradeforce.trader.buys import check_buy_options, buy_assets
from tradeforce.trader.sells import check_sell_options, sell_assets, sell_confirmed


class Trader:
    """_summary_"""

    def __init__(self, root):
        self.root = root
        self.config = root.config
        self.log = root.logging.getLogger(__name__)

        self.wallets = {}
        self.open_orders = []
        self.closed_orders = []
        self.gid = 10**9
        self.min_order_sizes = {}

        self.check_run_conditions()
        self.finalize_trading_config()

    ##################
    # Initial checks #
    ##################
    # TODO: Move checks to main thread -> relevant to sim as well
    def check_run_conditions(self):
        if (self.config.amount_invest_fiat is None) and (self.config.amount_invest_relative is None):
            sys.exit("[ERROR] Either 'amount_invest_fiat' or 'amount_invest_relative' must be set.")

        if (self.config.amount_invest_fiat is not None) and (self.config.amount_invest_relative is not None):
            self.log.info(
                "'amount_invest_fiat' and 'amount_invest_relative' are both set."
                + "'amount_invest_fiat' will be overwritten by 'amount_invest_relative' (relative to the budget)."
            )

    def finalize_trading_config(self):
        if self.config.amount_invest_relative is not None and self.config.budget > 0:
            self.config.amount_invest_fiat = np.round(self.config.budget * self.config.amount_invest_relative, 2)
        if self.config.buy_limit_strategy and self.config.budget > 0:
            self.config.asset_buy_limit = self.config.budget // self.config.amount_invest_fiat

    ####################
    # Order operations #
    ####################

    def new_order(self, order, order_type):
        order_obj = getattr(self, order_type)
        order_obj.append(order)
        if self.config.use_backend:
            db_response = self.root.backend.order_new(order.copy(), order_type)
            if not db_response:
                self.log.error("Backend DB insert order failed!")

    def edit_order(self, order, order_type):
        order_obj = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("buy_order_id") != order["buy_order_id"]]
        order_obj.append(order)

        if self.config.use_backend:
            db_response = self.root.backend.order_edit(order.copy(), order_type)
            if not db_response:
                self.log.error("Backend DB edit order failed!")

    def del_order(self, order, order_type):
        # Delete from internal mirror of DB
        order_obj = getattr(self, order_type)
        order_obj[:] = [o for o in order_obj if o.get("asset") != order["asset"]]
        if self.config.use_backend:
            db_response = self.root.backend.order_del(order.copy(), order_type)
            if not db_response:
                self.log.error("Backend DB delete order failed!")

    ##################
    # Getting orders #
    ##################

    def get_open_order(self, asset_order=None, asset=None):
        gid = None
        buy_order_id = None
        price_profit = None
        if asset_order is not None:
            asset_symbol = convert_symbol_str(
                asset_order.symbol, base_currency=self.config.base_currency, to_exchange=False
            )
            buy_order_id = asset_order.id
            gid = asset_order.gid
        else:
            asset_symbol = asset["asset"]
            buy_order_id = asset.get("buy_order_id", None)
            gid = asset.get("gid", None)
            price_profit = asset.get("price_profit", None)

        query = f"asset == '{asset_symbol}'"

        if buy_order_id is not None:
            query += f" and buy_order_id == {buy_order_id}"
        if gid is not None:
            query += f" and gid == {gid}"
        if price_profit is not None:
            query += f" and price_profit == {price_profit}"

        open_orders = []
        df_open_orders = pd.DataFrame(self.open_orders)
        if not df_open_orders.empty:
            open_orders = df_open_orders.query(query).to_dict("records")
        return open_orders

    def get_all_open_orders(self, raw=False):
        if raw:
            all_open_orders = self.open_orders
        else:
            all_open_orders = pd.DataFrame(self.open_orders)
        return all_open_orders

    def get_all_closed_orders(self, raw=False):
        if raw:
            all_closed_orders = self.closed_orders
        else:
            all_closed_orders = pd.DataFrame(self.closed_orders)
        return all_closed_orders

    #############################
    # Trader relevant functions #
    #############################

    async def update(self, latest_prices=None, timestamp=None):
        sell_options = check_sell_options(self.root, latest_prices, timestamp)
        if len(sell_options) > 0:
            await sell_assets(self.root, sell_options)
        buy_options = check_buy_options(self.root, latest_prices, timestamp)
        if len(buy_options) > 0:
            await buy_assets(self.root, buy_options)

    async def check_sold_orders(self):
        exchange_order_history = await self.root.exchange_api.get_order_history()
        sell_order_ids = self.get_all_open_orders()["sell_order_id"].to_list()
        sold_orders = [
            order
            for order in exchange_order_history
            if order["id"] in sell_order_ids and "EXECUTED" in order["order_status"]
        ]
        for sold_order in sold_orders:
            self.log.info(
                "Sold order of %s (id:%s gid:%s) has been converted to closed order.",
                sold_order["symbol"],
                sold_order["id"],
                sold_order["gid"],
            )
            sell_confirmed(self.root, sold_order)

    def set_budget(self, ws_wallet_snapshot):
        for wallet in ws_wallet_snapshot:
            if wallet.currency == self.config.base_currency:
                is_snapshot = wallet.balance_available is None
                base_currency_balance = wallet.balance if is_snapshot else wallet.balance_available
                self.config.budget = base_currency_balance
                self.root.backend.update_status({"budget": base_currency_balance})

    def get_profit(self):
        profit_fiat = np.round(sum(order["profit_fiat"] for order in self.closed_orders), 2)
        return profit_fiat

    async def get_min_order_sizes(self):
        self.min_order_sizes = await self.root.exchange_api.get_min_order_sizes()
