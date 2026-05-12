"""
pattern_engine.py — 12 candlestick patterns for CE/PE entry and exit.

Constraints:
  • ONLY the last 3 candles are used (constraint 6)
  • Volume is NEVER checked (constraint 7)

Pattern tiers:
  87% → Morning Doji Star, Evening Doji Star
  85% → Three White Soldiers, Three Black Crows
  83% → Bullish Engulfing, Bearish Engulfing
  82% → Morning Star, Evening Star
  81% → Piercing Line, Dark Cloud Cover
  80% → Hammer, Shooting Star

Exit behaviour:
  ≥83% (IMMEDIATE_EXIT_PATTERNS): exit on same candle — no confirmation needed
  80–82%: set pending_reversal flag; exit when SAR also reverses
"""
from __future__ import annotations
from typing import Optional, Tuple
from candle_builder import Candle
from logger_setup import get_module_logger

logger = get_module_logger("Patterns")

# ── Tier sets ─────────────────────────────────────────────────────────────────

IMMEDIATE_EXIT_PATTERNS: frozenset = frozenset({
    "Morning Doji Star",
    "Evening Doji Star",
    "Three White Soldiers",
    "Three Black Crows",
    "Bullish Engulfing",
    "Bearish Engulfing",
})

BULLISH_PATTERNS: frozenset = frozenset({
    "Morning Doji Star", "Three White Soldiers", "Bullish Engulfing",
    "Morning Star", "Piercing Line", "Hammer",
})

BEARISH_PATTERNS: frozenset = frozenset({
    "Evening Doji Star", "Three Black Crows", "Bearish Engulfing",
    "Evening Star", "Dark Cloud Cover", "Shooting Star",
})

# ── Candle helpers ────────────────────────────────────────────────────────────

def _body(c: Candle) -> float:
    return abs(c.close - c.open_price)

def _range(c: Candle) -> float:
    return c.high - c.low

def _upper_wick(c: Candle) -> float:
    return c.high - max(c.close, c.open_price)

def _lower_wick(c: Candle) -> float:
    return min(c.close, c.open_price) - c.low

def _is_doji(c: Candle, threshold: float = 0.1) -> bool:
    r = _range(c)
    return (_body(c) / r) <= threshold if r > 0 else True

# ── 3-candle patterns ─────────────────────────────────────────────────────────

def _morning_doji_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    return (
        c1.is_bearish()
        and _body(c1) > _range(c1) * 0.5
        and _is_doji(c2)
        and c2.high < c1.close
        and c3.is_bullish()
        and c3.close > (c1.open_price + c1.close) / 2
    )

def _evening_doji_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    return (
        c1.is_bullish()
        and _body(c1) > _range(c1) * 0.5
        and _is_doji(c2)
        and c2.low > c1.close
        and c3.is_bearish()
        and c3.close < (c1.open_price + c1.close) / 2
    )

def _three_white_soldiers(c1: Candle, c2: Candle, c3: Candle) -> bool:
    return (
        c1.is_bullish() and c2.is_bullish() and c3.is_bullish()
        and c2.open_price >= c1.close * 0.995
        and c3.open_price >= c2.close * 0.995
        and c2.close > c1.close
        and c3.close > c2.close
        and _body(c1) > _range(c1) * 0.5
        and _body(c2) > _range(c2) * 0.5
        and _body(c3) > _range(c3) * 0.5
    )

def _three_black_crows(c1: Candle, c2: Candle, c3: Candle) -> bool:
    return (
        c1.is_bearish() and c2.is_bearish() and c3.is_bearish()
        and c2.open_price <= c1.close * 1.005
        and c3.open_price <= c2.close * 1.005
        and c2.close < c1.close
        and c3.close < c2.close
        and _body(c1) > _range(c1) * 0.5
        and _body(c2) > _range(c2) * 0.5
        and _body(c3) > _range(c3) * 0.5
    )

def _morning_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    return (
        c1.is_bearish()
        and _body(c1) > _range(c1) * 0.5
        and _body(c2) < _body(c1) * 0.3
        and c3.is_bullish()
        and c3.close > (c1.open_price + c1.close) / 2
    )

def _evening_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    return (
        c1.is_bullish()
        and _body(c1) > _range(c1) * 0.5
        and _body(c2) < _body(c1) * 0.3
        and c3.is_bearish()
        and c3.close < (c1.open_price + c1.close) / 2
    )

# ── 2-candle patterns ─────────────────────────────────────────────────────────

def _bullish_engulfing(c1: Candle, c2: Candle) -> bool:
    return (
        c1.is_bearish()
        and c2.is_bullish()
        and c2.open_price <= c1.close
        and c2.close >= c1.open_price
        and _body(c2) > _body(c1)
    )

def _bearish_engulfing(c1: Candle, c2: Candle) -> bool:
    return (
        c1.is_bullish()
        and c2.is_bearish()
        and c2.open_price >= c1.close
        and c2.close <= c1.open_price
        and _body(c2) > _body(c1)
    )

def _piercing_line(c1: Candle, c2: Candle) -> bool:
    mid = (c1.open_price + c1.close) / 2
    return (
        c1.is_bearish()
        and c2.is_bullish()
        and c2.open_price < c1.low
        and mid < c2.close < c1.open_price
    )

def _dark_cloud_cover(c1: Candle, c2: Candle) -> bool:
    mid = (c1.open_price + c1.close) / 2
    return (
        c1.is_bullish()
        and c2.is_bearish()
        and c2.open_price > c1.high
        and c1.open_price < c2.close < mid
    )

# ── 1-candle patterns ─────────────────────────────────────────────────────────

def _hammer(c: Candle) -> bool:
    r = _range(c)
    return r > 0 and _lower_wick(c) >= 2 * _body(c) and _upper_wick(c) <= _body(c) * 0.3 and _body(c) > 0

def _shooting_star(c: Candle) -> bool:
    r = _range(c)
    return r > 0 and _upper_wick(c) >= 2 * _body(c) and _lower_wick(c) <= _body(c) * 0.3 and _body(c) > 0


# ── Pattern Engine ────────────────────────────────────────────────────────────

class PatternEngine:

    def scan(self, candles: list[Candle]) -> Tuple[Optional[str], str]:
        """
        Scan ≤3 candles for the strongest pattern present.
        Returns (pattern_name, 'bullish'|'bearish'|'none').
        Volume is NOT checked (constraint 7).
        """
        n = len(candles)
        if n == 0:
            return None, "none"

        # ── 3-candle (strongest first) ────────────────────────────────────────
        if n >= 3:
            c1, c2, c3 = candles[-3], candles[-2], candles[-1]

            if _morning_doji_star(c1, c2, c3):
                logger.info("▲ Morning Doji Star (87%) — BULLISH")
                return "Morning Doji Star", "bullish"

            if _three_white_soldiers(c1, c2, c3):
                logger.info("▲ Three White Soldiers (85%) — BULLISH")
                return "Three White Soldiers", "bullish"

            if _morning_star(c1, c2, c3):
                logger.info("▲ Morning Star (82%) — BULLISH")
                return "Morning Star", "bullish"

            if _evening_doji_star(c1, c2, c3):
                logger.info("▼ Evening Doji Star (87%) — BEARISH")
                return "Evening Doji Star", "bearish"

            if _three_black_crows(c1, c2, c3):
                logger.info("▼ Three Black Crows (85%) — BEARISH")
                return "Three Black Crows", "bearish"

            if _evening_star(c1, c2, c3):
                logger.info("▼ Evening Star (82%) — BEARISH")
                return "Evening Star", "bearish"

        # ── 2-candle ──────────────────────────────────────────────────────────
        if n >= 2:
            c1, c2 = candles[-2], candles[-1]

            if _bullish_engulfing(c1, c2):
                logger.info("▲ Bullish Engulfing (83%) — BULLISH")
                return "Bullish Engulfing", "bullish"

            if _piercing_line(c1, c2):
                logger.info("▲ Piercing Line (81%) — BULLISH")
                return "Piercing Line", "bullish"

            if _bearish_engulfing(c1, c2):
                logger.info("▼ Bearish Engulfing (83%) — BEARISH")
                return "Bearish Engulfing", "bearish"

            if _dark_cloud_cover(c1, c2):
                logger.info("▼ Dark Cloud Cover (81%) — BEARISH")
                return "Dark Cloud Cover", "bearish"

        # ── 1-candle ──────────────────────────────────────────────────────────
        c = candles[-1]
        if _hammer(c):
            logger.info("▲ Hammer (80%) — BULLISH")
            return "Hammer", "bullish"
        if _shooting_star(c):
            logger.info("▼ Shooting Star (80%) — BEARISH")
            return "Shooting Star", "bearish"

        return None, "none"

    def is_reversal_of(
        self, candles: list[Candle], open_direction: str
    ) -> Tuple[Optional[str], str]:
        """Returns pattern if it REVERSES the open position's direction."""
        pat, direction = self.scan(candles)
        if not pat:
            return None, "none"
        if open_direction == "bullish" and direction == "bearish":
            return pat, direction
        if open_direction == "bearish" and direction == "bullish":
            return pat, direction
        return None, "none"
