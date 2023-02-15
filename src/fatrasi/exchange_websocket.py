"""_summary_

Returns:
    _type_: _description_
"""

import traceback
import pandas as pd

from fatrasi.utils import convert_symbol_str
from fatrasi.trader_buys import buy_confirmed
from fatrasi.trader_sells import sell_confirmed


def check_timestamp_difference(start=None, end=None, freq="5min"):
    print(f"[DEBUG] Check delta between DB and WS timestamp of candles: {start} (DB) {end} (WS)")
    diff_range = (
        pd.date_range(
            start=pd.to_datetime(start, unit="ms", utc=True),
            end=pd.to_datetime(end, unit="ms", utc=True),
            freq=freq,
            inclusive="neither",
        ).asi8
        // 10**6
    )
    return diff_range


def convert_order_to_dict(order_obj):
    order_dict = {
        "symbol": order_obj.symbol,
        "gid": order_obj.gid,
        "mts_update": order_obj.mts_update,
        "price_avg": order_obj.price_avg,
        "amount_orig": order_obj.amount_orig,
    }
    return order_dict


class ExchangeWebsocket:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, fts=None):
        self.fts = fts
        self.config = fts.config

        self.bfx_api_priv = fts.api["bfx_api_priv"]
        self.bfx_api_pub = fts.api["bfx_api_pub"]

        self.ws_candle_cache = {}
        self.candle_cache_cap = 20
        self.asset_candle_subs = {}
        self.latest_candle_timestamp = 0
        self.current_candle_timestamp = 0
        self.last_candle_timestamp = 0
        self.prevent_race_condition_list = []
        self.ws_subs_finished = False
        self.is_set_last_candle_timestamp = False
        self.history_sync_patch_running = False

    ###################
    # Init websockets #
    ###################

    def ws_run(self):
        self.bfx_api_pub.ws.on("connected", self.ws_init_connection)
        self.bfx_api_pub.ws.on("new_candle", self.ws_new_candle)
        self.bfx_api_pub.ws.on("error", self.ws_error)
        self.bfx_api_pub.ws.on("subscribed", self.ws_is_subscribed)
        self.bfx_api_pub.ws.on("unsubscribed", self.ws_unsubscribed)
        self.bfx_api_pub.ws.on("status_update", self.ws_status)
        self.bfx_api_pub.ws.run()

    def ws_priv_run(self):
        if self.bfx_api_priv is not None:
            self.bfx_api_priv.ws.on("wallet_snapshot", self.ws_priv_wallet_snapshot)
            self.bfx_api_priv.ws.on("wallet_update", self.ws_priv_wallet_update)
            self.bfx_api_priv.ws.on("order_confirmed", self.ws_priv_order_confirmed)
            self.bfx_api_priv.ws.on("order_closed", self.ws_priv_order_closed)
            self.bfx_api_priv.ws.run()

    ######################
    # Round robin caches #
    ######################

    def prune_race_condition_prevention_cache(self):
        if len(self.prevent_race_condition_list) > 3:
            del self.prevent_race_condition_list[0]

    def prune_candle_cache(self):
        candles_timestamps = self.ws_candle_cache.keys()
        candle_cache_size = len(self.ws_candle_cache.keys())
        while candle_cache_size > self.candle_cache_cap:
            print(
                f"[DEBUG] Deleting {min(candles_timestamps)} from candle cache "
                + f"(max cap: {self.candle_cache_cap} candles)"
            )
            # delete oldest candle cache entry
            del self.ws_candle_cache[min(candles_timestamps)]
            candle_cache_size -= 1

    #############################
    # Public websocket channels #
    #############################

    def ws_error(self, ws_error):
        print(traceback.format_exc())
        print(f"[ERROR] ws_error: {ws_error}")

    def ws_is_subscribed(self, ws_subscribed):
        symbol = convert_symbol_str(ws_subscribed.symbol, to_exchange=False)
        self.asset_candle_subs[symbol] = ws_subscribed

    def ws_unsubscribed(self, ws_unsubscribed):
        print(f"[INFO] Unsubscribed: {ws_unsubscribed}")

    def ws_status(self, ws_status):
        # TODO: Handle if exchange goes down / maintanance
        # Check availability of exchange periodically ?!
        print(f"[WARNING] {ws_status}")

    async def ws_subscribe_candles(self, asset_list):
        asset_list_bfx = convert_symbol_str(
            asset_list, base_currency=self.config.base_currency, to_exchange=True, exchange=self.config.exchange
        )
        for symbol in asset_list_bfx:
            # asset_interval[:-2] to convert "5min" -> "5m"
            await self.bfx_api_pub.ws.subscribe_candles(symbol, self.config.asset_interval[:-2])
        self.ws_subs_finished = True

    async def ws_init_connection(self):
        self.latest_candle_timestamp = self.fts.market_history.get_local_candle_timestamp(position="latest")
        if self.latest_candle_timestamp == 0:
            self.latest_candle_timestamp = await self.fts.exchange_api.get_latest_remote_candle_timestamp(
                minus_delta=self.config.history_timeframe
            )
        self.is_set_last_candle_timestamp = False
        print("[INFO] Subscribing to channels..")
        await self.ws_subscribe_candles(self.fts.assets_list_symbols)

    async def ws_new_candle(self, candle):
        self.current_candle_timestamp = int(candle["mts"])
        symbol_converted = convert_symbol_str(
            candle["symbol"], base_currency=self.config.base_currency, to_exchange=False, exchange=self.config.exchange
        )
        current_asset = self.ws_candle_cache.get(self.current_candle_timestamp, {})
        current_asset[f"{symbol_converted}_o"] = candle["open"]
        current_asset[f"{symbol_converted}_c"] = candle["close"]
        current_asset[f"{symbol_converted}_h"] = candle["high"]
        current_asset[f"{symbol_converted}_l"] = candle["low"]
        current_asset[f"{symbol_converted}_v"] = candle["volume"]
        self.ws_candle_cache[self.current_candle_timestamp] = current_asset

        if not self.is_set_last_candle_timestamp:
            candles_timestamps = self.ws_candle_cache.keys()
            candle_cache_size = len(candles_timestamps)
            if candle_cache_size >= 2:
                self.is_set_last_candle_timestamp = True
                self.last_candle_timestamp = max(candles_timestamps)
                print(f"[DEBUG] last_candle_timestamp set to {self.last_candle_timestamp}")
                # Check sync of candle history. Patch if neccesary
                diff_range_candle_timestamps = check_timestamp_difference(
                    start=self.latest_candle_timestamp, end=self.last_candle_timestamp, freq=self.config.asset_interval
                )
                if len(diff_range_candle_timestamps) > 0:
                    timestamp_patch_history_start = min(diff_range_candle_timestamps)
                    timestamp_patch_history_end = max(diff_range_candle_timestamps)
                    if timestamp_patch_history_start != timestamp_patch_history_end:
                        self.history_sync_patch_running = True
                        print(
                            "[INFO] Patching out of sync history.."
                            + f" From {timestamp_patch_history_start} to {timestamp_patch_history_end}"
                        )
                        await self.fts.market_history.update(
                            start=timestamp_patch_history_start, end=timestamp_patch_history_end
                        )
                        self.history_sync_patch_running = False

        if (
            self.current_candle_timestamp not in self.prevent_race_condition_list
            and self.current_candle_timestamp > self.last_candle_timestamp
            and self.is_set_last_candle_timestamp
            and self.ws_subs_finished
        ):
            self.prevent_race_condition_list.append(self.current_candle_timestamp)
            print(
                "[INFO] Saving last candle into " + f"{self.config.backend} (timestamp: {self.last_candle_timestamp})"
            )
            candle_cache_size = len(self.ws_candle_cache.keys())
            candles_last_timestamp = self.ws_candle_cache.get(self.last_candle_timestamp, {})
            if not candles_last_timestamp:
                print(
                    f"[WARNING] Last websocket {self.config.asset_interval[:-2]} timestamp has no data from any asset."
                )
                # TODO: Trigger notification email?

            df_history_update = pd.DataFrame(candles_last_timestamp, index=[self.last_candle_timestamp])
            self.last_candle_timestamp = self.current_candle_timestamp
            df_history_update.index.name = "t"
            self.fts.backend.db_add_history(df_history_update)

            if self.fts.trader is not None:
                await self.fts.trader.check_sold_orders()
            if not self.history_sync_patch_running and not self.config.is_simulation and self.fts.trader is not None:
                await self.fts.trader.update()

            self.prune_candle_cache()
            self.prune_race_condition_prevention_cache()

            # TODO: Check exceptions
            # health_check_size = 10
            # check_result = self.check_ws_health(health_check_size)
            # if candle_cache_size >= health_check_size:
            #     print(f"[INFO] Result WS health check [based on {candle_cache_size} candle timestamps]:")
            #     print(
            #         f"Potentionally unhealthy: {check_result['unhealthy']}"
            #         if len(check_result["unhealthy"]) > 0
            #         else "All good"
            #     )
            # if len(check_result["not_subscribed"]) > 0:
            #     print(f"[WARNING] assets not subscribed: {check_result['not_subscribed']}")
            #     print("[INFO] Trying resub..")
            #     await self.ws_subscribe_candles(check_result["not_subscribed"])

    ##############################
    # Private websocket channels #
    ##############################

    def ws_priv_order_confirmed(self, ws_confirmed):
        print("order_confirmed", ws_confirmed)
        order_type = "buy" if ws_confirmed.amount_orig > 0 else "sell"
        if order_type == "sell":
            asset_symbol = convert_symbol_str(
                ws_confirmed.symbol, base_currency=self.config.base_currency, to_exchange=False
            )
            buy_order = {"asset": asset_symbol, "gid": ws_confirmed.gid}
            open_order = self.fts.trader.get_open_order(asset=buy_order)
            if len(open_order) > 0:
                open_order_edited = open_order[0]
                open_order_edited["sell_order_id"] = ws_confirmed.id
                self.fts.trader.edit_order(open_order_edited, "open_orders")
            else:
                print(f"[ERROR] Cannot find open order ({buy_order})")

    async def ws_priv_order_closed(self, ws_order_closed):
        print("order_closed", ws_order_closed)
        order_closed_and_filled = abs(abs(ws_order_closed.amount_orig) - abs(ws_order_closed.amount_filled)) < 0.0000001
        order_type = "buy" if ws_order_closed.amount_orig > 0 else "sell"
        if order_closed_and_filled:
            if order_type == "buy":
                await buy_confirmed(self.fts, ws_order_closed)
            if order_type == "sell":
                order_closed_dict = convert_order_to_dict(ws_order_closed)
                sell_confirmed(self.fts, order_closed_dict)

    async def ws_priv_wallet_snapshot(self, ws_wallet_snapshot):
        print("wallet_snapshot", ws_wallet_snapshot)
        self.fts.trader.set_budget(ws_wallet_snapshot)
        self.fts.trader.finalize_trading_config()
        self.fts.backend.db_sync_trader_state()
        await self.fts.trader.get_min_order_sizes()

    def ws_priv_wallet_update(self, ws_wallet_update):
        if ws_wallet_update.currency == self.config.base_currency:
            self.fts.trader.set_budget([ws_wallet_update])
        else:
            self.fts.trader.wallets[ws_wallet_update.currency] = ws_wallet_update

    # def check_ws_health(self, health_check_size=10):
    #     health_check_size *= -1
    #     latest_candle_timestamps = sorted(self.ws_candle_cache.keys())[health_check_size:]
    #     healthy_assets = {}

    #     for candle_timestamp in latest_candle_timestamps:
    #         assets = get_col_names(self.ws_candle_cache[candle_timestamp])
    #         # collect how often an asset candle appears within health_check_size
    #         for asset in assets:
    #             current_asset = healthy_assets.get(asset, 0)
    #             healthy_assets[asset] = current_asset + 1
    #     # for now we only check if a candle of an asset was received within health_check_size
    #     # if no candle was received within health_check_size the asset is considered "unhealthty"
    #     # if at least one candle was received, the asset is considered "healthy"
    #     healthy_assets_list = list(healthy_assets.keys())
    #     unhealthy_assets = np.setdiff1d(self.fts.assets_list_symbols, healthy_assets_list)
    #     not_subscribed_anymore = []

    #     # check if still subscribed
    #     for asset in unhealthy_assets:
    #         asset_is_subbed = False
    #         asset_sub = self.asset_candle_subs.get(asset, None)
    #         if asset_sub is not None:
    #             asset_is_subbed = asset_sub.is_subscribed()
    #         if not asset_is_subbed:
    #             not_subscribed_anymore.append(asset)

    #     return {"healthy": healthy_assets_list,
    #            "unhealthy": unhealthy_assets,
    #            "not_subscribed": not_subscribed_anymore}
