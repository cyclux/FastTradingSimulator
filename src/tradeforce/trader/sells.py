"""_summary_
"""
import numpy as np
from tradeforce.utils import calc_fee, convert_symbol_str


def check_sell_options(root, latest_prices=None, timestamp=None):
    # TODO: Reconsider the "ok_to_sell logic". Maybe adapt/edit price_profit to check if it was already reduced?
    # Also consider future feature of "dynamic price decay"
    sell_options = []
    portfolio_performance = {}
    if latest_prices is None:
        df_latest_prices = root.market_history.get_market_history(latest_candle=True, metrics=["c"], uniform_cols=True)
        timestamp = df_latest_prices.index[0]
        latest_prices = df_latest_prices.to_dict("records")[0]
    open_orders = root.trader.get_all_open_orders(raw=True)
    for open_order in open_orders:
        symbol = open_order["asset"]
        price_current = latest_prices.get(symbol, 0)
        price_buy = open_order["price_buy"]
        price_profit = open_order["price_profit"]

        if np.isnan(price_current):
            price_current = 0.0
        current_profit_ratio = price_current / price_buy
        portfolio_performance[symbol] = np.round(current_profit_ratio, 2)
        time_since_buy = (timestamp - open_order["timestamp_buy"]) / 1000 / 60 // 5

        ok_to_sell = (
            time_since_buy > root.config.hold_time_limit and current_profit_ratio >= root.config.profit_ratio_limit
        )
        # TODO: is_simulation == dry_run?!
        if root.config.is_simulation:
            if price_current >= price_profit or ok_to_sell:
                # check plausibility and prevent false logic
                # profit gets a max plausible threshold
                if price_current / price_profit > 1.2:
                    price_current = price_profit
                sell_option = {"asset": symbol, "price_sell": price_current}
                sell_options.append(sell_option)
        else:
            if ok_to_sell:
                sell_option = {
                    "asset": symbol,
                    "price_sell": price_current,
                    "gid": open_order["gid"],
                    "buy_order_id": open_order["buy_order_id"],
                }
                sell_options.append(sell_option)
    if not root.config.is_simulation:
        root.log.info("Current portfolio performance: %s", portfolio_performance)

    return sell_options


async def sell_assets(root, sell_options):
    for sell_option in sell_options:
        open_order = root.trader.get_open_order(asset=sell_option)
        if len(open_order) < 1:
            continue

        if root.config.is_simulation:
            # TODO: Replace with sell_confirmed() ?
            closed_order = open_order[0].copy()
            closed_order["price_sell"] = sell_option["price_sell"]
            sell_volume_crypto, _, sell_fee_fiat = calc_fee(
                root.config,
                closed_order["buy_volume_crypto"],
                closed_order["price_sell"],
                order_type="sell",
            )
            closed_order["sell_fee_fiat"] = sell_fee_fiat
            closed_order["sell_volume_crypto"] = sell_volume_crypto
            closed_order["sell_volume_fiat"] = (sell_volume_crypto * closed_order["price_sell"]) - sell_fee_fiat
            closed_order["profit_fiat"] = closed_order["sell_volume_fiat"] - closed_order["amount_invest_fiat"]
            root.trader.new_order(closed_order, "closed_orders")
            root.trader.del_order(open_order[0], "open_orders")

            new_budget = float(np.round(root.config.budget + closed_order["sell_volume_fiat"], 2))
            # TODO: Trader does not have "budget" ?
            root.backend.update_status({"budget": new_budget})
        else:
            # Adapt sell price
            volatility_buffer = 0.00000005
            sell_order = {
                "sell_order_id": open_order[0]["sell_order_id"],
                "gid": open_order[0]["gid"],
                "asset": open_order[0]["asset"],
                "price": sell_option["price_sell"],
                "amount": open_order[0]["buy_volume_crypto"] - volatility_buffer,
            }
            order_result_ok = await root.exchange_api.order("sell", sell_order, update_order=True)
            if order_result_ok:
                root.log.info(
                    "Sell price of %s has been changed from %s to %s.",
                    sell_order["asset"],
                    open_order[0]["price_buy"],
                    sell_order["price"],
                )


async def submit_sell_order(root, open_order):
    volatility_buffer = 0.00000002
    sell_order = {
        "asset": open_order["asset"],
        "price": open_order["price_profit"],
        "amount": open_order["buy_volume_crypto"] - volatility_buffer,
        "gid": open_order["gid"],
    }
    exchange_result_ok = await root.exchange_api.order("sell", sell_order)
    if not exchange_result_ok:
        root.log.error("Sell order execution failed! -> %s", str(sell_order))


def sell_confirmed(root, sell_order):
    root.log.debug("sell_order confirmed: %s", sell_order)
    asset_symbol = convert_symbol_str(sell_order["symbol"], base_currency=root.config.base_currency, to_exchange=False)
    open_order = root.trader.get_open_order(asset={"asset": asset_symbol, "gid": sell_order["gid"]})
    if len(open_order) > 0:
        closed_order = open_order[0].copy()
        closed_order["sell_timestamp"] = sell_order["mts_update"]
        closed_order["price_sell"] = sell_order["price_avg"]
        sell_volume_crypto = abs(sell_order["amount_orig"])
        sell_volume_crypto_incl_fee, _, closed_order["sell_fee_fiat"] = calc_fee(
            root.config, sell_volume_crypto, sell_order["price_avg"], order_type="sell"
        )
        sell_volume_fiat_incl_fee = sell_volume_crypto_incl_fee * sell_order["price_avg"]
        closed_order["sell_volume_fiat"] = np.round(sell_volume_fiat_incl_fee - closed_order["sell_fee_fiat"], 3)

        closed_order["profit_fiat"] = closed_order["sell_volume_fiat"] - closed_order["amount_invest_fiat"]
        root.trader.new_order(closed_order, "closed_orders")
        root.trader.del_order(open_order[0], "open_orders")
    else:
        root.log.error("Could not find order to sell: %s", sell_order)
