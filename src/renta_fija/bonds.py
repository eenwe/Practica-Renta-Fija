from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .data_loading import DEFAULT_VALUATION_DATE


def year_fraction(d1: datetime, d2: datetime, basis: int = 365) -> float:
    """Fracción de año simple ACT/basis."""
    return (d2 - d1).days / basis


def discount_from_curve(date: datetime, curva: pd.DataFrame, val_date: datetime = DEFAULT_VALUATION_DATE) -> float:
    """Devuelve el factor de descuento interpolado de la curva para una fecha dada."""
    t = year_fraction(val_date, date)
    if t <= 0:
        return 1.0
    return float(np.interp(t, curva["t"].values, curva["Discount"].values))


def build_cashflows(
    row: pd.Series,
    val_date: datetime = DEFAULT_VALUATION_DATE,
    notional: float = 100.0,
):
    """
    Genera fechas de cupón y flujos (cupón + principal).
    Supone cupón fijo y frecuencia en 'Coupon Frequency'.
    """
    maturity = row["Maturity"]
    first_coupon = row["First Coupon Date"]
    freq = int(row["Coupon Frequency"])
    coupon_rate = row["Coupon"] / 100.0  # de % a decimal
    coupon_per_period = notional * coupon_rate / freq

    if pd.isna(maturity) or pd.isna(first_coupon) or freq <= 0:
        return [], []

    months = 12 // freq
    dates = []
    dt = maturity
    while dt > val_date and dt >= first_coupon:
        dates.append(dt)
        dt = dt - relativedelta(months=months)

    dates = sorted(dates)
    if not dates:
        return [], []

    cashflows = [coupon_per_period] * len(dates)
    cashflows[-1] += notional  # añadimos el principal en el último flujo

    return dates, cashflows


def model_price_from_curve(
    row: pd.Series,
    curva: pd.DataFrame,
    val_date: datetime = DEFAULT_VALUATION_DATE,
    notional: float = 100.0,
) -> float:
    """Precio teórico del bono usando la curva de descuento."""
    cf_dates, cf_amounts = build_cashflows(row, val_date, notional)
    if not cf_dates:
        return np.nan

    dfs = np.array([discount_from_curve(d, curva, val_date) for d in cf_dates])
    pv = np.sum(np.array(cf_amounts) * dfs)
    return float(pv)


def price_given_yield(
    y: float,
    cf_dates,
    cf_amounts,
    val_date: datetime = DEFAULT_VALUATION_DATE,
) -> float:
    """Precio del bono dado un yield constante y flujos de caja."""
    ts = np.array([year_fraction(val_date, d) for d in cf_dates])
    dfs = 1 / (1 + y) ** ts
    return float(np.sum(np.array(cf_amounts) * dfs))


def yield_to_maturity(
    market_price: float,
    cf_dates,
    cf_amounts,
    val_date: datetime = DEFAULT_VALUATION_DATE,
) -> float:
    """Calcula la TIR de un bono por búsqueda numérica."""
    if not cf_dates:
        return np.nan

    def f(y):
        return price_given_yield(y, cf_dates, cf_amounts, val_date) - market_price

    try:
        ytm = brentq(f, -0.05, 0.5)
    except ValueError:
        return np.nan
    return float(ytm)


def duration_and_convexity(
    market_price: float,
    cf_dates,
    cf_amounts,
    y: float,
    val_date: datetime = DEFAULT_VALUATION_DATE,
):
    """Devuelve duración Macaulay, duración modificada y convexidad."""
    ts = np.array([year_fraction(val_date, d) for d in cf_dates])
    dfs = 1 / (1 + y) ** ts
    pv = np.array(cf_amounts) * dfs

    D_mac = np.sum(ts * pv) / market_price
    D_mod = D_mac / (1 + y)
    C = np.sum(ts * (ts + 1) * pv / (1 + y) ** 2) / market_price

    return float(D_mac), float(D_mod), float(C)
