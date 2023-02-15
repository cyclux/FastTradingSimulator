"""_summary_

Returns:
    _type_: _description_
"""

import numpy as np
import pandas as pd
from fatrasi.utils import get_col_names


###########################
# Calculate asset metrics #
###########################


def add_pct_change_cols(assets_history_asset, inplace=True):
    if inplace:
        for column in assets_history_asset.columns:
            assets_history_asset.loc[:, f"{column}_pct"] = assets_history_asset[column].pct_change() + 1
        return None

    cols_pct_change = {}
    for column in assets_history_asset.columns:
        cols_pct_change[f"{column}_pct"] = assets_history_asset.loc[:, column].pct_change() + 1

    return pd.DataFrame(cols_pct_change)


def aggregate_history(df_input, agg_timeframe="1h"):
    amount_five_min_intervals = pd.Timedelta(agg_timeframe).value // 10**9 // 60 // 5
    agg_func_map = {
        "o": lambda row: row[0],
        "h": np.nanmax,
        "l": np.nanmin,
        "c": lambda row: row[-1],
        "v": np.nansum,
    }
    forward_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=amount_five_min_intervals)
    relevant_assets_agg_list = []
    for candle_type, func in agg_func_map.items():
        candle_type_cols = [col for col in df_input.columns if col[-1] == candle_type]
        relevant_assets_agg_list.append(
            df_input.loc[:, candle_type_cols]
            .rolling(window=forward_indexer, step=amount_five_min_intervals, min_periods=1)
            .apply(func, raw=True)
        )
    return pd.concat(relevant_assets_agg_list, axis=1)[df_input.columns]


def get_asset_performance_metrics(df_input):
    assets_volume_usd = get_volume_usd(df_input)
    amount_candles = get_amount_candles(df_input)
    candle_density = get_candle_density(df_input)
    asset_volatility = get_asset_volatility(df_input)
    asset_metrics = pd.DataFrame([assets_volume_usd, amount_candles, candle_density, asset_volatility]).T
    asset_metrics.columns = ["vol_usd", "amount_candles", "candle_density", "volatility"]
    return asset_metrics


async def get_init_relevant_assets(fts, capped=-1):
    # 34 days ~ 10000 candles limit
    print("[INFO] Analyzing market for relevant assets..")
    init_market_history = await fts.market_updater_api.update_market_history(init_timespan="34days")
    df_relevant_assets_metrics = get_asset_performance_metrics(init_market_history).query(
        "amount_candles > 2000 & candle_density < 500"
    )
    relevant_asset_symbols = df_relevant_assets_metrics.sort_values("amount_candles", ascending=False).index
    if capped > 0:
        relevant_asset_symbols = relevant_asset_symbols[:capped]
    print("[INFO] Market analysis finished!")
    return {
        "assets": list(relevant_asset_symbols),
        "metrics": df_relevant_assets_metrics,
        "data": init_market_history,
    }


def get_volume_usd(df_input):
    volume_usd = {}
    assets = get_col_names(df_input.columns)
    for asset in assets:
        volume_usd[asset] = int(np.sum(df_input[f"{asset}_c"] * df_input[f"{asset}_v"]))
    return pd.Series(volume_usd).sort_values()


def get_amount_candles(df_input):
    candles_vol = get_col_names(df_input.columns, specific_col="v")
    amount_candles = df_input[candles_vol].count()
    amount_candles.index = get_col_names(amount_candles.index)
    return amount_candles.sort_values()


def get_candle_density(df_input):
    candles_vol = get_col_names(df_input.columns, specific_col="v")
    df_vol = df_input[candles_vol]
    df_candle_density = {}
    for col in df_vol.columns:
        asset_index_not_nan = df_vol[col][pd.notna(df_vol[col])].index
        asset_index_not_nan /= 1000
        df_candle_density[col] = int(np.diff(asset_index_not_nan).mean())
    series_candle_density = pd.Series(df_candle_density)
    series_candle_density.index = get_col_names(series_candle_density.index)
    return series_candle_density


def get_asset_volatility(df_input):
    candles_open = get_col_names(df_input.columns, specific_col="o")
    assets_agg_open = aggregate_history(df_input, agg_timeframe="1h").loc[:, candles_open]
    assets_agg_open_pct = add_pct_change_cols(assets_agg_open, inplace=False)
    return pd.Series(
        np.nanstd(assets_agg_open_pct, axis=0), index=get_col_names(assets_agg_open_pct.columns)
    ).sort_values()


def get_asset_buy_performance(fts, history_window=150, timestamp=None):
    start = -1 * history_window
    end = None
    idx_type = "iloc"
    if timestamp is not None:
        idx_type = "loc"
        start = timestamp - (history_window * 300000)
        end = timestamp
    market_window_pct_change = fts.market_history.get_market_history(
        start=start,
        end=end,
        idx_type=idx_type,
        pct_change=True,
        pct_as_factor=False,
        metrics=["o"],
        fill_na=True,
        uniform_cols=True,
    )
    if len(market_window_pct_change) < history_window:
        buy_performance = None
    else:
        buy_performance = market_window_pct_change.sum()
    return buy_performance
