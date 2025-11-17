import pandas as pd
from datetime import datetime

# Fecha de valoración por defecto: coincide con la última fecha de precios
DEFAULT_VALUATION_DATE = datetime.strptime("01/10/2025", "%d/%m/%Y")


def load_universo(path: str, valuation_date: datetime = DEFAULT_VALUATION_DATE) -> pd.DataFrame:
    """
    Carga el fichero universo.csv y hace una limpieza básica:
    - Convierte columnas de fecha a datetime
    - Calcula el time-to-maturity (Ttm_years)
    - Filtra solo bonos en EUR y con vencimiento futuro
    """
    univ = pd.read_csv(path, sep=";")

    # Columnas que son fechas en el universo
    date_cols = [
        "Maturity",
        "First Coupon Date",
        "Penultimate Coupon Date",
        "Issue date",
        "Next Call Date",
    ]

    for c in date_cols:
        if c in univ.columns:
            univ[c] = pd.to_datetime(
                univ[c],
                format="%d/%m/%Y",
                dayfirst=True,
                errors="coerce",
            )

    # Tiempo a vencimiento en años (desde la fecha de valoración)
    univ["Ttm_years"] = (univ["Maturity"] - valuation_date).dt.days / 365

    # Filtro sencillo: solo bonos en EUR y con vencimiento futuro
    if "Ccy" in univ.columns:
        univ = univ[(univ["Ccy"] == "EUR") & (univ["Ttm_years"] > 0)].copy()

    return univ


def load_prices_universo_long(path: str) -> pd.DataFrame:
    """
    Carga el fichero precios_historicos_universo.csv y lo devuelve en formato 'largo':
    columnas: ISIN, Date, Price

    La primera columna del fichero viene como 'XS0161488498 Corp', etc.
    Aquí limpiamos el ISIN quedándonos con la primera palabra.
    """
    prices_univ = pd.read_csv(path, sep=";", na_values=["#N/D"])

    # Renombramos la primera columna a ISIN_raw
    first_col = prices_univ.columns[0]
    prices_univ.rename(columns={first_col: "ISIN_raw"}, inplace=True)

    # ISIN limpio = primera "palabra" antes del espacio
    prices_univ["ISIN"] = prices_univ["ISIN_raw"].str.split().str[0]

    # El resto de columnas son fechas (precios diarios)
    price_cols = [c for c in prices_univ.columns if c not in ["ISIN_raw", "ISIN"]]

    # Pasamos al formato largo: una fila por (ISIN, fecha)
    prices_long = prices_univ.melt(
        id_vars="ISIN",
        value_vars=price_cols,
        var_name="Date",
        value_name="Price",
    )

    prices_long["Date"] = pd.to_datetime(
        prices_long["Date"],
        format="%d/%m/%Y",
        dayfirst=True,
        errors="coerce",
    )
    prices_long["Price"] = pd.to_numeric(prices_long["Price"], errors="coerce")

    return prices_long


def add_market_price(
    univ_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    valuation_date: datetime = DEFAULT_VALUATION_DATE,
) -> pd.DataFrame:
    """
    Añade una columna 'MarketPrice' al universo con el precio de cierre
    en la fecha de valoración. Si falta, usa la columna 'Price' original.
    """
    today_prices = (
        prices_long[prices_long["Date"] == valuation_date]
        .set_index("ISIN")["Price"]
    )

    out = univ_df.merge(
        today_prices.rename("MarketPrice"),
        on="ISIN",
        how="left",
    )

    if "Price" in out.columns:
        out["MarketPrice"] = out["MarketPrice"].fillna(out["Price"])

    return out


def load_curva_estr(path: str, valuation_date: datetime = DEFAULT_VALUATION_DATE) -> pd.DataFrame:
    """
    Carga la curva ESTR y añade la columna 't':
    - 't' es el tiempo en años desde la fecha de valoración
    Asegura también que 'Zero Rate' y 'Discount' son numéricos.
    """
    curva = pd.read_csv(path, sep=";")

    curva["Date"] = pd.to_datetime(
        curva["Date"],
        format="%d/%m/%Y",
        dayfirst=True,
        errors="coerce",
    )
    curva = curva.sort_values("Date").reset_index(drop=True)

    curva["t"] = (curva["Date"] - valuation_date).dt.days / 365
    curva["t"] = curva["t"].clip(lower=0.0)

    if "Zero Rate" in curva.columns:
        curva["Zero Rate"] = pd.to_numeric(curva["Zero Rate"], errors="coerce")
    if "Discount" in curva.columns:
        curva["Discount"] = pd.to_numeric(curva["Discount"], errors="coerce")

    return curva
