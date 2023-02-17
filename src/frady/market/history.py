"""_summary_
Returns:
    _type_: _description_
"""
import sys
import os
from pathlib import Path
import pandas as pd
from frady.utils import ms_to_ns, get_col_names, ns_to_ms, get_time_minus_delta
from frady.market.metrics import get_init_relevant_assets


def get_pct_change(df_history, as_factor=True):
    df_history_pct = df_history.pct_change()
    if as_factor:
        df_history_pct += 1
    df_history_pct.columns = [f"{col}_pct" for col in df_history_pct.columns]
    df_history = pd.concat([df_history, df_history_pct], axis=1, copy=False)
    return df_history_pct


def get_timestamp_intervals(start, end):
    # 50000min == 10000 * 5min -> 10000 max len @ bitfinex api
    timestamp_intervals = [(start, end)]
    if start and end:
        timestamp_list = ns_to_ms(
            pd.date_range(start=ms_to_ns(start), end=ms_to_ns(end), tz="UTC", freq="50000min").asi8
        ).tolist()
        if timestamp_list[-1] != end:
            timestamp_list.append(end)
        timestamp_intervals = list(zip(timestamp_list, timestamp_list[1:]))

    return timestamp_intervals


# TODO: Optimize data types in DBs
# def set_column_type(assets_history_asset, columns_type_int):
#     # Seperate loop because dependend on completion of pct_change
#     for column in assets_history_asset.columns:
#         if column in columns_type_int:
#             assets_history_asset.loc[:, column] = assets_history_asset[column].values.astype(np.int32)
#         else:
#             assets_history_asset.loc[:, column] = assets_history_asset[column].values.astype(np.float32)


def df_fill_na(df_input):
    hist_update_cols = df_input.columns
    volume_cols = hist_update_cols[[("_v" in col) for col in hist_update_cols]]
    ohlc_cols = hist_update_cols[
        [("_o" in col) or ("_h" in col) or ("_l" in col) or ("_c" in col) for col in hist_update_cols]
    ]
    df_input.loc[:, volume_cols] = df_input.loc[:, volume_cols].fillna(0)
    df_input.loc[:, ohlc_cols] = df_input.loc[:, ohlc_cols].fillna(method="ffill")
    df_input.loc[:, ohlc_cols] = df_input.loc[:, ohlc_cols].fillna(method="bfill")


class MarketHistory:
    """Fetch market history from disk and store in DataFrame

    Returns:
        _type_: _description_
    """

    def __init__(self, fts=None, path_current=None, path_history_dumps=None):

        self.fts = fts
        self.config = fts.config
        self.df_market_history = None

        # Set paths
        self.path_current = (
            Path(os.path.dirname(os.path.abspath(__file__))) if path_current is None else Path(path_current)
        )
        self.path_feather = (
            self.path_current / "data/inputs" if path_history_dumps is None else self.path_current / path_history_dumps
        )
        Path(self.path_feather).mkdir(parents=True, exist_ok=True)
        self.feather_filename = (
            f"{self.config.exchange}_{self.config.base_currency}_{self.config.mongo_collection}.arrow"
        )

    async def load_history(self):
        print(f"[INFO] Loading history via {self.config.load_history_via}")
        if self.config.load_history_via == "mongodb":
            if self.fts.backend.is_collection_new:
                sys.exit(
                    f"[ERROR] MongoDB collection '{self.config.mongo_collection}' does not exist. "
                    + "Choose correct collection or get initial market history via 'API' or local storage 'feather'."
                )
            cursor = self.fts.backend.mongo_exchange_coll.find({}, sort=[("t", 1)], projection={"_id": False})
            self.df_market_history = pd.DataFrame(list(cursor))

        elif self.config.load_history_via == "feather":
            self.df_market_history = pd.read_feather(self.path_feather / self.feather_filename)
            self.df_market_history.set_index("t", inplace=True)
            self.df_market_history.sort_index(inplace=True)

        elif self.config.load_history_via == "api":
            if self.fts.assets_list_symbols is not None:
                end = await self.fts.exchange_api.get_latest_remote_candle_timestamp()
                start = get_time_minus_delta(end, delta=self.config.history_timeframe)["timestamp"]
            else:
                relevant_assets = await get_init_relevant_assets(self.fts, capped=self.config.relevant_assets_cap)
                self.fts.assets_list_symbols = relevant_assets["assets"]
                filtered_assets = [
                    f"{asset}_{metric}" for asset in relevant_assets["assets"] for metric in ["o", "h", "l", "c", "v"]
                ]
                await self.update(history_data=relevant_assets["data"][filtered_assets])

                latest_local_candle_timestamp = self.get_local_candle_timestamp(position="latest")
                start_time = get_time_minus_delta(latest_local_candle_timestamp, delta=self.config.history_timeframe)
                start = start_time["timestamp"]
                first_local_candle_timestamp = self.get_local_candle_timestamp(position="first")

                end_time = get_time_minus_delta(first_local_candle_timestamp, delta=self.config.candle_interval)
                end = end_time["timestamp"]
            print(
                f"[INFO] Fetching {self.config.history_timeframe} ({start} - {end}) of market history "
                + f"from {len(self.fts.assets_list_symbols)} assets"
            )
            if start < end:
                await self.update(start=start, end=end)

        else:
            sys.exit(
                "[ERROR] load_history_via = '{self.load_history_via}' does not exist. "
                + "Available are 'api', 'mongodb', 'feather' and 'csv'."
            )

        try:
            self.df_market_history.set_index("t", inplace=True)
        except (KeyError, ValueError, TypeError):
            pass

        if self.config.load_history_via in ("api", "mongodb"):
            if self.config.dump_to_feather is True:
                self.dump_to_feather()

        if self.fts.assets_list_symbols is None:
            self.get_asset_symbols(updated=True)

        if self.fts.backend.sync_check_needed:
            self.fts.backend.db_sync_check()

        if self.config.update_history is True:
            start = self.get_local_candle_timestamp(position="latest", offset=1)
            end = await self.fts.exchange_api.get_latest_remote_candle_timestamp()
            if start < end:
                await self.update(start=start, end=end)
            else:
                print(f"[INFO] Market history is already uptodate ({start})")

        if self.config.check_db_consistency is True:
            self.fts.backend.check_db_consistency()

        amount_assets = len(self.fts.assets_list_symbols)
        print(f"[INFO] {amount_assets} assets from {self.config.exchange} loaded via {self.config.load_history_via}")

        return self.df_market_history

    def dump_to_feather(self):
        self.df_market_history.reset_index(drop=False).to_feather(self.path_feather / self.feather_filename)
        print(f"[INFO] Assets dumped via feather to: {self.path_feather / self.feather_filename}")

    def get_market_history(
        self,
        assets=None,
        start=None,
        end=None,
        from_list=None,
        latest_candle=False,
        idx_type="loc",
        metrics=None,
        pct_change=False,
        pct_as_factor=True,
        fill_na=False,
        uniform_cols=False,
    ):

        metrics = metrics if metrics else ["o", "h", "l", "c", "v"]
        assets = assets if assets else self.get_asset_symbols(updated=True)

        if latest_candle:
            idx_type = "iloc"
            start = -1

        if from_list is not None:
            idx_type = None

        if start and isinstance(start, str):
            latest_local_candle_timestamp = self.get_local_candle_timestamp(position="latest")
            try:
                start = (pd.to_datetime(latest_local_candle_timestamp, unit="ms", utc=True) - pd.Timedelta(start)).value

            except ValueError as exc:
                raise ValueError(
                    f"ERROR: {start} is not a valid pandas time unit abbreviation!\n"
                    + "For reference check: "
                    + "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
                ) from exc
            else:
                start = ns_to_ms(start)

        # Contruct list of columns to filter
        assets_to_filter = [f"{asset}_{metric}" for asset in assets for metric in metrics]
        self.df_market_history.sort_index(inplace=True)
        if fill_na and not self.fts.backend.is_filled_na:
            df_fill_na(self.df_market_history)
            self.fts.backend.is_filled_na = True

        if idx_type == "loc":
            df_market_history = self.df_market_history.loc[start:end, assets_to_filter]
        if idx_type == "iloc":
            assets_to_filter_idx = [self.df_market_history.columns.tolist().index(asset) for asset in assets_to_filter]
            df_market_history = self.df_market_history.iloc[start:end, assets_to_filter_idx]
        if from_list is not None:
            df_market_history = self.df_market_history.loc[from_list]

        if pct_change:
            df_market_history = get_pct_change(df_market_history, as_factor=pct_as_factor)

        if uniform_cols and len(metrics) == 1:
            df_market_history.columns = get_col_names(df_market_history)
        return df_market_history

    async def update(self, history_data=None, start=None, end=None):
        assets_status = None
        if history_data is not None:
            df_history_update = history_data
        else:
            timestamp_intervals = get_timestamp_intervals(start, end)
            df_history_update_list = []
            for interval in timestamp_intervals:
                print(f"[INFO] Fetching history interval: {interval}")
                i_start = interval[0]
                i_end = interval[1]
                df_history_update_interval = await self.fts.market_updater_api.update_market_history(
                    start=i_start, end=i_end
                )
                if df_history_update_interval is None:
                    return assets_status
                df_history_update_list.append(df_history_update_interval)
            df_history_update = pd.concat(df_history_update_list, axis=0)
            df_history_update = df_history_update.iloc[~df_history_update.index.duplicated()]
        self.fts.backend.db_add_history(df_history_update)
        return assets_status

    ###########
    # Getters #
    ###########
    def get_asset_symbols(self, updated=False):
        """Returns a list of all asset symbols

        Returns:
            list: str
        """
        if updated:
            self.fts.assets_list_symbols = get_col_names(self.df_market_history.columns)
        return self.fts.assets_list_symbols

    def get_local_candle_timestamp(self, position="latest", skip=0, offset=0):
        idx = (-1 - skip) if position == "latest" else (0 + skip)
        latest_candle_timestamp = 0
        if self.df_market_history is not None:
            latest_candle_timestamp = int(self.df_market_history.index[idx])
        elif self.config.backend == "mongodb" and not self.fts.backend.is_collection_new:
            sort_id = -1 if position == "latest" else 1
            cursor = self.fts.backend.mongo_exchange_coll.find({}, sort=[("t", sort_id)], skip=skip, limit=1)
            latest_candle_timestamp = int(cursor[0]["t"])

        latest_candle_timestamp += offset * 300000
        return latest_candle_timestamp
