"""_summary_

Returns:
    _type_: _description_
"""

from time import sleep, process_time
from functools import reduce

import pandas as pd

from frady.utils import (
    get_now,
    get_timedelta,
    get_time_minus_delta,
    get_df_datetime_index,
)


def append_market_history_update(assets_hist_update, assets_candles, symbol_current):
    current_asset = assets_hist_update.get(symbol_current, [])
    current_asset.extend(
        [
            {
                "t": i[0],
                symbol_current + "_o": i[1],
                symbol_current + "_c": i[2],
                symbol_current + "_h": i[3],
                symbol_current + "_l": i[4],
                symbol_current + "_v": i[5],
            }
            for i in assets_candles
        ]
    )
    return current_asset


class MarketUpdater:
    """_summary_"""

    def __init__(self, root=None):
        self.root = root
        self.config = root.config

    async def get_timeframe(self, start=None, end=None):
        candle_freq_in_ms = get_timedelta(self.config.candle_interval)["timestamp"]
        timeframe = {}

        timeframe["start_timestamp"] = (
            start if start else self.root.market_history.get_local_candle_timestamp(position="latest")
        )
        if timeframe["start_timestamp"] == 0:
            timeframe["start_timestamp"] = await self.root.exchange_api.get_latest_remote_candle_timestamp(
                minus_delta=self.root.market_history.history_timeframe
            )
        elif not start:
            # only needed when fetching as update to existing history
            timeframe["start_timestamp"] += candle_freq_in_ms

        timeframe["start_datetime"] = pd.to_datetime(timeframe["start_timestamp"], unit="ms", utc=True)

        if end:
            timeframe["end_timestamp"] = end
            timeframe["end_datetime"] = pd.to_datetime(timeframe["end_timestamp"], unit="ms", utc=True)
        else:
            now = get_now()
            timeframe["end_datetime"] = now["datetime"]
            timeframe["end_timestamp"] = now["timestamp"]

        ms_additional_wait = get_timedelta("2min")["timestamp"]
        ms_until_wait_over = timeframe["end_timestamp"] - (timeframe["start_timestamp"] + ms_additional_wait)
        return {"timeframe": timeframe, "ms_until_wait_over": ms_until_wait_over}

    async def fetch_market_history(self, timeframe):
        market_history_update = {}
        for symbol in self.root.assets_list_symbols:
            time_start = process_time()
            candle_update = await self.root.exchange_api.get_public_candles(
                symbol=symbol,
                base_currency=self.config.base_currency,
                timestamp_start=timeframe["start_timestamp"],
                timestamp_end=timeframe["end_timestamp"],
            )
            if len(candle_update) > 0:
                market_history_update[symbol] = append_market_history_update(
                    market_history_update, candle_update, symbol
                )
                print(f"[INFO] Fetched updated history for {symbol} ({len(candle_update)} candles)")

            time_elapsed_difference = 1 - (process_time() - time_start)
            if time_elapsed_difference > 0:
                sleep(time_elapsed_difference)
        return market_history_update

    def convert_market_history_to_df(self, history_update, timeframe=None):
        if not history_update:
            print("[WARNING] No market update, cannot create dataframe")
            return None
        history_df_list = []
        if timeframe is not None:
            history_df_list.append(get_df_datetime_index(timeframe, freq=self.config.candle_interval))

        for asset_hist in history_update.values():
            if len(asset_hist) > 0:
                df_hist = pd.DataFrame(asset_hist)
                df_hist.sort_values("t", inplace=True, ascending=True)
                history_df_list.append(df_hist)

        df_market_history_update = reduce(
            lambda df_left, df_right: pd.merge_asof(df_left, df_right, on="t", direction="nearest", tolerance=60000),
            history_df_list,
        )
        df_market_history_update.set_index("t", inplace=True)
        return df_market_history_update

    async def update_market_history(self, start=None, end=None, init_timespan=None):
        if init_timespan is not None:
            latest_remote_candle_timestamp = await self.root.exchange_api.get_latest_remote_candle_timestamp()
            start = get_time_minus_delta(timestamp=latest_remote_candle_timestamp, delta=init_timespan)["timestamp"]

        if self.root.assets_list_symbols is None:
            self.root.assets_list_symbols = await self.root.exchange_api.get_active_assets()

        timeframe = await self.get_timeframe(start=start, end=end)

        if timeframe["ms_until_wait_over"] > 0:
            market_history_update = await self.fetch_market_history(timeframe["timeframe"])
            if len(market_history_update) > 1:
                df_market_history_update = self.convert_market_history_to_df(
                    market_history_update, timeframe=timeframe["timeframe"]
                )
            else:
                print(
                    f"[INFO] No market update since {pd.to_timedelta(timeframe['ms_until_wait_over'], unit='ms')}. "
                    + "Maybe exchange down?"
                )
        else:
            print(
                f"[INFO] Update request too early. Frequency is {self.config.candle_interval}. "
                + f"Wait at least {int(abs(timeframe['ms_until_wait_over']) / 1000 // 60)} min for next try."
            )
        return df_market_history_update
