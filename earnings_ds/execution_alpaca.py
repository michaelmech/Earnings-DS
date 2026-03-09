import json
import time

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

_TERMINAL = {"filled", "canceled", "rejected", "expired"}

def _positions_map(trading: TradingClient) -> dict[str, float]:
    return {str(p.symbol): float(p.qty) for p in trading.get_all_positions()}


def _parse_apierror_payload(e: APIError) -> dict:
    try:
        return json.loads(e.args[0])
    except Exception:
        try:
            return json.loads(str(e))
        except Exception:
            return {}


def _get_symbol_orders(trading: TradingClient, symbol: str, *, limit: int = 500):
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        symbols=[symbol],
        nested=True,   # helps surface bracket legs on some accounts
        limit=limit,
    )
    return trading.get_orders(filter=req)


def _iter_order_tree(order):
    yield order
    for leg in (getattr(order, "legs", None) or []):
        yield leg


def _side(o) -> str:
    return str(getattr(o, "side", "")).lower()


def _status(o) -> str:
    return str(getattr(o, "status", "")).lower()


def _is_live(o) -> bool:
    return _status(o) not in _TERMINAL


def _cancel_ids(trading: TradingClient, ids: list[str]) -> int:
    n = 0
    for oid in ids:
        try:
            trading.cancel_order_by_id(oid)
            n += 1
        except Exception:
            pass
    return n


def _cancel_all_live_sells_for_symbol(trading: TradingClient, symbol: str) -> int:
    n = 0
    for parent in _get_symbol_orders(trading, symbol):
        for o in _iter_order_tree(parent):
            if _is_live(o) and _side(o) == "sell":
                try:
                    trading.cancel_order_by_id(o.id)
                    n += 1
                except Exception:
                    pass
    return n


def _wait_no_live_sells(trading: TradingClient, symbol: str, *, timeout_s: float = 12.0, poll_s: float = 0.25) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        live_sell = False
        for parent in _get_symbol_orders(trading, symbol):
            for o in _iter_order_tree(parent):
                if _is_live(o) and _side(o) == "sell":
                    live_sell = True
                    break
            if live_sell:
                break
        if not live_sell:
            return True
        time.sleep(poll_s)
    return False


def _is_insufficient_available(payload: dict) -> bool:
    return payload.get("code") == 40310000 and "insufficient qty available" in str(payload.get("message", "")).lower()


def _is_trading_halt_market_reject(payload: dict) -> bool:
    msg = str(payload.get("message", "")).lower()
    return payload.get("code") == 42210000 and "trading halt" in msg and "limit order" in msg


def _passes_spread_gate(
    row: pd.Series,
    *,
    spread_multiplier: float | None,
    current_price_col: str,
    current_spread_col: str,
) -> tuple[bool, dict]:
    """Check whether spread gate allows order execution."""
    if spread_multiplier is None:
        return True, {}

    if current_price_col not in row.index or current_spread_col not in row.index:
        return False, {
            "reason": "missing_spread_inputs",
            "required_columns": [current_price_col, current_spread_col],
        }

    px = row.get(current_price_col)
    spread = row.get(current_spread_col)

    if pd.isna(px) or pd.isna(spread):
        return False, {
            "reason": "nan_spread_inputs",
            "current_price": px,
            "current_spread": spread,
        }

    px = float(px)
    spread = float(spread)
    threshold_value = float(spread_multiplier) * px

    if threshold_value > spread:
        return True, {
            "threshold_value": threshold_value,
            "current_price": px,
            "current_spread": spread,
        }

    return False, {
        "reason": "spread_gate_blocked",
        "threshold_value": threshold_value,
        "current_price": px,
        "current_spread": spread,
    }


def rebalance_to_targets(
    df_targets: pd.DataFrame,
    *,
    target_qty_col: str = "shares",
    take_profit_col: str = "tp_price",
    stop_loss_col: str = "sl_price",
    use_market_entries: bool = True,
    entry_price_col: str = "entry",
    tif: TimeInForce = TimeInForce.GTC,
    paper: bool = True,

    # behavior switches
    cancel_blocking_orders: bool = True,
    halt_fallback_to_limit: bool = True,
    halt_limit_price_col: str = "entry",  # use your entry column as limit on halts
    spread_multiplier: float | None = None,
    current_price_col: str = "current_price",
    current_spread_col: str = "current_spread",
):
    required = {target_qty_col, take_profit_col, stop_loss_col}
    missing = sorted(required - set(df_targets.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # net duplicates by ticker
    df = df_targets.copy()
    df.index = df.index.astype(str)
    df = (
        df.sort_index()
          .groupby(level=0)
          .agg({
              target_qty_col: "sum",
              take_profit_col: "last",
              stop_loss_col: "last",
              **({entry_price_col: "last"} if entry_price_col in df.columns else {}),
          })
    )

    trading = TradingClient(api_key, api_secret, paper=paper)
    current = _positions_map(trading)
    results = []

    for symbol, row in df.iterrows():
        target_qty = float(row[target_qty_col])
        cur_qty = float(current.get(symbol, 0.0))
        delta = target_qty - cur_qty

        if abs(delta) < 1e-9:
            results.append({"symbol": symbol, "cur_qty": cur_qty, "target_qty": target_qty, "delta": 0.0, "action": "none"})
            continue

        pass_gate, gate_info = _passes_spread_gate(
            row,
            spread_multiplier=spread_multiplier,
            current_price_col=current_price_col,
            current_spread_col=current_spread_col,
        )
        if not pass_gate:
            results.append({"symbol": symbol, "cur_qty": cur_qty, "target_qty": target_qty, "delta": delta,
                            "action": "skip_spread_gate", **gate_info})
            continue

        # -------- SELL delta (reduce) --------
        if delta < 0:
            sell_qty = abs(delta)

            def submit_sell_market():
                return trading.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=sell_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )

            def submit_sell_limit(limit_px: float):
                return trading.submit_order(
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=sell_qty,
                        side=OrderSide.SELL,
                        limit_price=limit_px,
                        time_in_force=TimeInForce.DAY,
                    )
                )

            try:
                order = submit_sell_market()
                results.append({"symbol": symbol, "action": "sell_delta", "order_id": getattr(order, "id", None), "status": getattr(order, "status", None)})
                continue

            except APIError as e:
                payload = _parse_apierror_payload(e)

                # trading halt -> place limit instead (uses your entry column as the limit price)
                if halt_fallback_to_limit and _is_trading_halt_market_reject(payload):
                    if halt_limit_price_col not in row or pd.isna(row.get(halt_limit_price_col, None)):
                        results.append({"symbol": symbol, "action": "skip_sell_halt_no_limit_price", "error": payload})
                        continue
                    order = submit_sell_limit(float(row[halt_limit_price_col]))
                    results.append({"symbol": symbol, "action": "sell_delta_limit_halt", "limit_price": float(row[halt_limit_price_col]),
                                    "order_id": getattr(order, "id", None), "status": getattr(order, "status", None)})
                    continue

                # insufficient available -> cancel blockers then retry once
                if cancel_blocking_orders and _is_insufficient_available(payload):
                    related = payload.get("related_orders") or []
                    canceled_related = _cancel_ids(trading, related)

                    # fallback: if related_orders incomplete, cancel any remaining live sells for symbol
                    canceled_fallback = _cancel_all_live_sells_for_symbol(trading, symbol)

                    _wait_no_live_sells(trading, symbol, timeout_s=15.0)

                    try:
                        order = submit_sell_market()
                        results.append({"symbol": symbol, "action": "sell_delta_after_cancel",
                                        "canceled_related": canceled_related, "canceled_fallback": canceled_fallback,
                                        "order_id": getattr(order, "id", None), "status": getattr(order, "status", None)})
                        continue
                    except APIError as e2:
                        results.append({"symbol": symbol, "action": "skip_sell_still_blocked",
                                        "canceled_related": canceled_related, "canceled_fallback": canceled_fallback,
                                        "error": _parse_apierror_payload(e2)})
                        continue

                raise

        # -------- BUY delta (increase) with bracket --------
        if delta > 0:
            tp = float(row[take_profit_col])
            sl = float(row[stop_loss_col])

            # if you want “always limit entries”, set use_market_entries=False
            if use_market_entries:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=delta,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                )
            else:
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=delta,
                    side=OrderSide.BUY,
                    limit_price=float(row[entry_price_col]),
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                )

            order = trading.submit_order(req)
            results.append({"symbol": symbol, "action": "buy_delta_bracket",
                            "order_id": getattr(order, "id", None), "status": getattr(order, "status", None)})

    return pd.DataFrame(results).set_index("symbol")


def submit_brackets_from_df(
    df: pd.DataFrame,
    *,
    qty_col: str = "qty",
    entry_price_col: str = "entry_price",      # used if you want LIMIT entries; can be None for market entries
    take_profit_col: str = "take_profit",
    stop_loss_col: str = "stop_loss",
    use_market_entries: bool = True,
    tif: TimeInForce = TimeInForce.GTC,
    paper: bool = True,
    spread_multiplier: float | None = None,
    current_price_col: str = "current_price",
    current_spread_col: str = "current_spread",
):
    """
    Assumes ticker symbol is df.index (e.g., df.index = ["AAPL","MSFT",...])
    Expects prices already in df columns (tp/sl). Sends a BRACKET order per row.
    """
    required = {qty_col, take_profit_col, stop_loss_col}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    if (not use_market_entries) and (entry_price_col not in df.columns):
        raise ValueError(f"use_market_entries=False requires '{entry_price_col}' column")

    # creds: set these env vars (recommended)

    # paper base_url is handled by alpaca-py via paper=True
    trading = TradingClient(api_key, api_secret, paper=paper)

    results = []
    for symbol, row in df.iterrows():
        qty = float(row[qty_col])
        tp = float(row[take_profit_col])
        sl = float(row[stop_loss_col])

        pass_gate, gate_info = _passes_spread_gate(
            row,
            spread_multiplier=spread_multiplier,
            current_price_col=current_price_col,
            current_spread_col=current_spread_col,
        )
        if not pass_gate:
            results.append({
                "symbol": str(symbol),
                "qty": qty,
                "entry_type": "market" if use_market_entries else "limit",
                "take_profit": tp,
                "stop_loss": sl,
                "status": "skipped",
                "action": "skip_spread_gate",
                **gate_info,
            })
            continue

        if qty <= 0:
            raise ValueError(f"{symbol}: qty must be > 0 (got {qty})")

        # Build the parent order (entry) + attached legs
        common_kwargs = dict(
            symbol=str(symbol),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=tif,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp),
            stop_loss=StopLossRequest(stop_price=sl),
        )

        if use_market_entries:
            req = MarketOrderRequest(**common_kwargs)
        else:
            entry_price = float(row[entry_price_col])
            req = LimitOrderRequest(limit_price=entry_price, **common_kwargs)

        order = trading.submit_order(req)
        results.append(
            {
                "symbol": str(symbol),
                "submitted_order_id": getattr(order, "id", None),
                "qty": qty,
                "entry_type": "market" if use_market_entries else "limit",
                "take_profit": tp,
                "stop_loss": sl,
                "status": getattr(order, "status", None),
            }
        )

    return pd.DataFrame(results).set_index("symbol")


