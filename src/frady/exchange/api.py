"""_summary_

Returns:
    _type_: _description_
"""

import numpy as np
import pandas as pd
from frady.utils import convert_symbol_str, ns_to_ms


class ExchangeAPI:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, root=None):
        self.root = root
        self.config = root.config

        self.bfx_api_priv = root.api["bfx_api_priv"]
        self.bfx_api_pub = root.api["bfx_api_pub"]

    #####################
    # REST API - Public #
    #####################

    async def get_active_assets(self, consider_exclude_list=True):
        bfx_active_symbols = await self.bfx_api_pub.rest.fetch("conf/", params="pub:list:pair:exchange")
        symbols_base_currency = convert_symbol_str(
            bfx_active_symbols,
            to_exchange=False,
            base_currency=self.config.base_currency,
            exchange=self.config.exchange,
        )
        if consider_exclude_list:
            assets_list_symbols = list(
                np.setdiff1d(symbols_base_currency, self.config.assets_excluded, assume_unique=True)
            )
        else:
            assets_list_symbols = symbols_base_currency
        return assets_list_symbols

    async def get_asset_stats(self, symbols):
        bfx_asset_stats = await self.bfx_api_pub.rest.get_public_tickers(symbols)
        bfx_asset_stats_formatted = {}
        for asset_stat in bfx_asset_stats:
            bfx_asset_stats_formatted[asset_stat[0]] = {
                "BID": asset_stat[1],
                "BID_SIZE": asset_stat[2],
                "ASK": asset_stat[3],
                "ASK_SIZE": asset_stat[4],
                "DAILY_CHANGE": asset_stat[5],
                "DAILY_CHANGE_PERC": asset_stat[6],
                "LAST_PRICE": asset_stat[7],
                "VOLUME": asset_stat[8],
                "HIGH": asset_stat[9],
                "LOW": asset_stat[10],
            }
        return bfx_asset_stats_formatted

    async def get_public_candles(
        self, symbol=None, base_currency=None, timestamp_start=None, timestamp_end=None, candle_type="hist"
    ):
        symbol_bfx = convert_symbol_str(
            symbol, to_exchange=True, base_currency=base_currency, exchange=self.config.exchange
        )

        candles = await self.bfx_api_pub.rest.get_public_candles(
            symbol_bfx, start=timestamp_start, end=timestamp_end, section=candle_type, limit=10000, tf="5m", sort=1
        )
        return candles

    async def get_latest_remote_candle_timestamp(self, minus_delta=None):
        indicator_assets = ["BTC", "ETH", "XRP", "ADA", "LTC"]

        latest_candle_timestamp_pool = []
        for indicator in indicator_assets:
            latest_exchange_candle = await self.get_public_candles(
                symbol=indicator, base_currency=self.config.base_currency, candle_type="last"
            )
            latest_candle_timestamp_pool.append(int(latest_exchange_candle[0]))
        latest_candle_timestamp = max(latest_candle_timestamp_pool)

        if minus_delta:
            latest_candle = pd.Timestamp(latest_candle_timestamp, unit="ms", tz="UTC") - pd.to_timedelta(minus_delta)
            latest_candle_timestamp = ns_to_ms(latest_candle.value)
        return latest_candle_timestamp

    async def get_min_order_sizes(self):
        bfx_asset_infos = await self.bfx_api_pub.rest.fetch("conf/", params="pub:info:pair")
        asset_symbols = self.root.market_history.get_asset_symbols()
        all_asset_symbols = convert_symbol_str(
            asset_symbols, base_currency="USD", with_trade_prefix=False, to_exchange=True
        )
        all_asset_symbols_info = [
            asset for asset in bfx_asset_infos[0] if asset[0][-3:] == "USD" and asset[0] in all_asset_symbols
        ]
        asset_min_order_sizes = {
            convert_symbol_str(asset[0], to_exchange=False): float(asset[1][3]) for asset in all_asset_symbols_info
        }
        return asset_min_order_sizes

    ######################
    # REST API - PRIVATE #
    ######################

    async def order(self, order_type, order_payload, update_order=False):
        exchange_result = False
        bfx_symbol = convert_symbol_str(
            order_payload["asset"], base_currency=self.config.base_currency, to_exchange=True
        )
        if order_type == "sell":
            order_amount = -1 * order_payload["amount"]
        else:
            order_amount = order_payload["amount"]
        try:
            if update_order:
                exchange_response = await self.bfx_api_priv.rest.submit_update_order(
                    order_payload["sell_order_id"], price=order_payload["price"], amount=order_amount
                )
            else:
                market_type_current = "EXCHANGE FOK" if order_type == "buy" else "EXCHANGE LIMIT"
                exchange_response = await self.bfx_api_priv.rest.submit_order(
                    symbol=bfx_symbol,
                    price=order_payload["price"],
                    amount=order_amount,
                    market_type=market_type_current,
                    gid=order_payload["gid"],
                )
            exchange_result = exchange_response.is_success()
        # Cannot specify Exception as bfx API does not provide more
        except Exception as exc:  # pylint: disable=broad-except
            print(exc)
        return exchange_result

    async def get_order_history(self, raw=False):
        order_history = await self.bfx_api_priv.rest.post("auth/r/orders/hist", data={"limit": 2500}, params="")
        if not raw:
            order_history = [
                {
                    "id": order[0],
                    "gid": order[1],
                    "symbol": order[3],
                    "mts_create": order[4],
                    "mts_update": order[5],
                    "amount": order[6],
                    "amount_orig": order[7],
                    "type": order[8],
                    "order_status": order[13],
                    "price": order[16],
                    "price_avg": order[17],
                }
                for order in order_history
            ]
        return order_history

    async def get_active_orders(self, symbol):
        bfx_active_orders = await self.bfx_api_priv.rest.get_active_orders(symbol)
        return bfx_active_orders

    async def cancel_order(self, cid):
        bfx_cancel_order = await self.bfx_api_priv.rest.submit_cancel_order(cid)
        return bfx_cancel_order

    async def get_wallets(self):
        bfx_wallets = await self.bfx_api_priv.rest.get_wallets()
        return bfx_wallets

    async def get_wallet_deposit_address(self, wallet=None, method=None, renew=0):
        bfx_deposit_address = await self.bfx_api_priv.rest.get_wallet_deposit_address(
            wallet=wallet, method=method, renew=renew
        )
        return bfx_deposit_address
