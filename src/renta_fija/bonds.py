from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .data_loading import DEFAULT_VALUATION_DATE


def year_fraction(d1: datetime, d2: datetime, basis: int = 365) -> float:
    """
    Fracción de año ACT/basis entre dos fechas.
    Por ejemplo, basis=365 → (d2 - d1).days / 365
    """
    return (d2 - d1).days / basis


def discount_from_curve(
    date: datetime,
    curva: pd.DataFrame,
    val_date: datetime = DEFAULT_VALUATION_DATE,
) -> float:
    """
    Devuelve el factor de descuento interpolado de la curva para una fecha dada.
    Usa la columna 't' y 'Discount' de la curva ESTR.
    """
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
    Genera fechas de cupón y flujos (cupón + principal) futuros.
    Supone:
    - Cupón fijo en la columna 'Coupon' (en % anual)
    - Frecuencia en 'Coupon Frequency' (1 = anual, 2 = semestral, etc.)
    - 'First Coupon Date' y 'Maturity' están informados

    Devuelve:
    - lista de fechas de flujo (cf_dates)
    - lista de importes de flujo (cf_amounts)
    """
    maturity = row["Maturity"]
    first_coupon = row["First Coupon Date"]
    freq = int(row["Coupon Frequency"])
    coupon_rate = row["Coupon"] / 100.0
    coupon_per_period = notional * coupon_rate / freq

    if pd.isna(maturity) or pd.isna(first_coupon) or freq <= 0:
        return [], []

    months = 12 // freq
    dates = []
    dt = maturity

    # Generamos cupones hacia atrás desde el vencimiento
    while dt > val_date and dt >= first_coupon:
        dates.append(dt)
        dt = dt - relativedelta(months=months)

    dates = sorted(dates)
    if not dates:
        return [], []

    cashflows = [coupon_per_period] * len(dates)
    # Añadimos el principal al último flujo (vencimiento)
    cashflows[-1] += notional

    return dates, cashflows


def model_price_from_curve(
    row: pd.Series,
    curva: pd.DataFrame,
    val_date: datetime = DEFAULT_VALUATION_DATE,
    notional: float = 100.0,
) -> float:
    """
    Precio teórico del bono usando la curva de descuento (curva ESTR).
    """
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
    """
    Precio del bono dado un yield constante y flujos de caja.
    Descuento discreto anual: 1 / (1 + y)^t
    """
    ts = np.array([year_fraction(val_date, d) for d in cf_dates])
    dfs = 1 / (1 + y) ** ts
    return float(np.sum(np.array(cf_amounts) * dfs))


def yield_to_maturity(
    market_price: float,
    cf_dates,
    cf_amounts,
    val_date: datetime = DEFAULT_VALUATION_DATE,
) -> float:
    """
    Calcula la TIR (yield to maturity) del bono.
    Es el y tal que precio_teórico(y) = precio_de_mercado.
    """
    if not cf_dates:
        return np.nan

    def f(y):
        return price_given_yield(y, cf_dates, cf_amounts, val_date) - market_price

    try:
        ytm = brentq(f, -0.05, 0.5)  # buscamos la raíz entre -5% y 50%
    except ValueError:
        return np.nan

    return float(ytm)


def rf_zero_rate_at_maturity(
    row: pd.Series,
    curva: pd.DataFrame,
    val_date: datetime = DEFAULT_VALUATION_DATE,
) -> float:
    """
    Tipo 'risk-free' (zero rate) correspondiente al plazo del bono.
    Interpolamos la columna 'Zero Rate' de la curva ESTR.
    """
    T = year_fraction(val_date, row["Maturity"])
    if T <= 0:
        return np.nan

    zero_rates = curva.dropna(subset=["Zero Rate"])
    return float(
        np.interp(
            T,
            zero_rates["t"].values,
            (zero_rates["Zero Rate"] / 100.0).values,  # pasamos de % a decimal
        )
    )


def duration_and_convexity(
    market_price: float,
    cf_dates,
    cf_amounts,
    y: float,
    val_date: datetime = DEFAULT_VALUATION_DATE,
):
    """
    Calcula:
    - Duración Macaulay
    - Duración modificada
    - Convexidad (versión discreta)
    """
    ts = np.array([year_fraction(val_date, d) for d in cf_dates])
    dfs = 1 / (1 + y) ** ts
    pv = np.array(cf_amounts) * dfs

    D_mac = np.sum(ts * pv) / market_price
    D_mod = D_mac / (1 + y)
    C = np.sum(ts * (ts + 1) * pv / (1 + y) ** 2) / market_price

    return float(D_mac), float(D_mod), float(C)


def compute_full_risk_measures(
    row: pd.Series,
    curva: pd.DataFrame,
    val_date: datetime = DEFAULT_VALUATION_DATE,
) -> pd.Series:
    """
    Calcula para un bono (una fila del DataFrame 'univ'):
    - YTM       (TIR)
    - rf_rate   (tipo libre de riesgo a su vencimiento)
    - spread    (YTM - rf_rate)
    - Dur_Mac   (duración Macaulay)
    - Dur_Mod   (duración modificada)
    - Convexity (convexidad)
    """
    cf_dates, cf_amounts = build_cashflows(row, val_date)
    if not cf_dates:
        return pd.Series(
            {
                "YTM": np.nan,
                "rf_rate": np.nan,
                "spread": np.nan,
                "Dur_Mac": np.nan,
                "Dur_Mod": np.nan,
                "Convexity": np.nan,
            }
        )

    price = row["MarketPrice"]
    ytm = yield_to_maturity(price, cf_dates, cf_amounts, val_date)
    rf = rf_zero_rate_at_maturity(row, curva, val_date)

    if pd.isna(ytm) or pd.isna(rf):
        spread = np.nan
    else:
        spread = ytm - rf

    if pd.isna(ytm):
        D_mac = D_mod = C = np.nan
    else:
        D_mac, D_mod, C = duration_and_convexity(price, cf_dates, cf_amounts, ytm, val_date)

    return pd.Series(
        {
            "YTM": ytm,
            "rf_rate": rf,
            "spread": spread,
            "Dur_Mac": D_mac,
            "Dur_Mod": D_mod,
            "Convexity": C,
        }
    )
