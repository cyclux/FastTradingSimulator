"""_summary_
"""

from asyncio import sleep as asyncio_sleep
import numpy as np
import pandas as pd
from frady.utils import convert_symbol_str
from frady.market.metrics import get_asset_buy_performance
from frady.trader.sells import submit_sell_order


def get_significant_digits(num, digits):
    return round(num, digits - int(np.floor(np.log10(abs(num)))) - 1)


def check_buy_options(fts, latest_prices=None, timestamp=None):
    buy_options = []
    if latest_prices is None:
        df_latest_prices = fts.market_history.get_market_history(latest_candle=True, metrics=["o"], uniform_cols=True)
        latest_prices = df_latest_prices.to_dict("records")[0]
        if timestamp is None:
            timestamp = df_latest_prices.index[0]
    buy_performance = get_asset_buy_performance(fts, history_window=fts.config.window, timestamp=timestamp)
    if buy_performance is not None:
        buy_condition = (buy_performance >= fts.config.buy_opportunity_factor_min) & (
            buy_performance <= fts.config.buy_opportunity_factor_max
        )
        buy_options = buy_performance[buy_condition]
        df_buy_options = pd.DataFrame({"perf": buy_options, "price": latest_prices}).dropna()
        if fts.config.prefer_performance == "negative":
            df_buy_options = df_buy_options.sort_values(by="perf", ascending=True)
        if fts.config.prefer_performance == "positive":
            df_buy_options = df_buy_options.sort_values(by="perf", ascending=False)
        if fts.config.prefer_performance == "center":
            df_buy_options.loc[:, "perf"] = np.absolute(df_buy_options["perf"] - fts.config.buy_opportunity_factor)
            df_buy_options = df_buy_options.sort_values(by="perf")

        df_buy_options.reset_index(names=["asset"], inplace=True)
        buy_options = df_buy_options.to_dict("records")

    amount_buy_options = len(buy_options)
    if amount_buy_options > 0:
        buy_options_print = [
            f"{buy_option['asset']} [perf:{np.round(buy_option['perf'], 2)}, price: {buy_option['price']}]"
            for buy_option in buy_options
        ]
        print(
            f"[INFO] {amount_buy_options} potential asset{'s' if len(buy_options) > 1 else ''} to buy:",
            *buy_options_print,
        )
    else:
        print("[INFO] Currently no potential assets to buy.")
    return buy_options


async def buy_assets(fts, buy_options):
    compensate_rate_limit = bool(len(buy_options) > 9)
    assets_out_of_funds_to_buy = []
    assets_max_amount_bought = []
    for asset in buy_options:
        asset_symbol = asset["asset"]
        # TODO: Make possible to have multiple orders of same asset
        if asset_symbol in fts.config.assets_excluded:
            print(f"[INFO] Asset on blacklist. Will not buy {asset}")
            continue
        asset_open_orders = fts.trader.get_open_order(asset=asset)
        if len(asset_open_orders) > 0:
            assets_max_amount_bought.append(asset_symbol)
            continue

        # Add 1% margin for BUY LIMIT order
        asset["price"] *= 1.015
        buy_amount_crypto = get_significant_digits(fts.config.amount_invest_fiat / asset["price"], 9)

        min_order_size = fts.trader.min_order_sizes.get(asset_symbol, 0)
        if min_order_size > buy_amount_crypto:
            print(
                f"[INFO] Adapting buy_amount_crypto ({buy_amount_crypto}) "
                + f"of {asset_symbol} to min_order_size ({min_order_size})"
            )
            buy_amount_crypto = min_order_size * 1.02

        if fts.config.budget < fts.config.amount_invest_fiat:
            assets_out_of_funds_to_buy.append(asset_symbol)
            continue

        buy_order = {
            "asset": asset_symbol,
            "gid": fts.trader.gid,
            "price": asset["price"],
            "amount": buy_amount_crypto,
        }
        print("[INFO] Executing buy order:", buy_order)

        if fts.config.is_simulation:
            new_budget = float(np.round(fts.config.budget - fts.config.amount_invest_fiat, 2))
            # TODO: Trader does not have "budget" ?
            fts.backend.update_status({"budget": new_budget})
        else:
            exchange_result_ok = await fts.exchange_api.order("buy", buy_order)
            fts.trader.gid += 1
            fts.backend.update_status({"gid": fts.trader.gid})
            if not exchange_result_ok:
                # TODO: Send notification about this event!
                print(f"[ERROR] Buy order execution failed! -> {buy_order}")
            if compensate_rate_limit:
                await asyncio_sleep(0.8)
    amount_assets_out_of_funds = len(assets_out_of_funds_to_buy)
    amount_assets_max_bought = len(assets_max_amount_bought)
    if amount_assets_out_of_funds > 0:
        print(
            f"[INFO] {amount_assets_out_of_funds} asset{'s' if amount_assets_out_of_funds > 1 else ''}"
            + " out of funds to buy "
            + f"(${np.round(fts.config.budget, 2)} < ${fts.config.amount_invest_fiat}):",
            *assets_out_of_funds_to_buy,
        )
    if amount_assets_max_bought > 0:
        print(
            f"[INFO] {amount_assets_max_bought} asset{'s' if amount_assets_max_bought > 1 else ''}"
            + f" {'have' if amount_assets_max_bought > 1 else 'has'} reached max amount to buy:",
            *assets_max_amount_bought,
        )


async def buy_confirmed(fts, buy_order):
    asset_price_profit = get_significant_digits(buy_order.price * fts.config.profit_factor, 5)
    asset_symbol = convert_symbol_str(buy_order.symbol, base_currency=fts.config.base_currency, to_exchange=False)
    buy_volume_fiat = float(np.round(fts.config.amount_invest_fiat - buy_order.fee, 5))
    # Wait until balance is registered by websocket into self.wallets
    await asyncio_sleep(10)
    buy_volume_crypto = fts.trader.wallets[asset_symbol].balance_available
    open_order = {
        "trader_id": fts.config.trader_id,
        "buy_order_id": buy_order.id,
        "gid": buy_order.gid,
        "timestamp_buy": int(buy_order.mts_create),
        "asset": asset_symbol,
        "base_currency": fts.config.base_currency,
        # TODO: "performance": asset["perf"],
        "price_buy": buy_order.price,
        "price_profit": asset_price_profit,
        "amount_invest_fiat": fts.config.amount_invest_fiat,
        "buy_volume_fiat": buy_volume_fiat,
        "buy_fee_fiat": buy_order.fee,
        "buy_volume_crypto": buy_volume_crypto,
    }
    fts.trader.new_order(open_order, "open_orders")
    if not fts.config.is_simulation:
        await submit_sell_order(fts, open_order)
