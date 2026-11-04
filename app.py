"""MVP-приложение для прогноза спроса и планирования закупок.

Запуск: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Demand Studio — прогноз спроса",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_COLUMNS = {"Дата", "Товар", "Количество"}
SALES_COLUMNS = ["Дата", "Товар", "Количество"]
INVENTORY_COLUMNS = ["Товар", "Остаток", "Срок поставки"]


def create_demo_sales() -> pd.DataFrame:
    """Создаёт детерминированную историю продаж за последние 35 дней."""
    today = date.today()
    products = {
        "Кроссовки Nike Air": (12, 2),
        "Футболка Adidas": (21, 6),
        "Кепка Puma": (8, 3),
    }
    rows: list[dict[str, object]] = []
    for day_offset in range(35, -1, -1):
        sale_date = today - timedelta(days=day_offset)
        weekday_factor = 1.25 if sale_date.weekday() in (5, 6) else 1.0
        for product, (base, phase) in products.items():
            # Небольшая повторяющаяся вариация делает демонстрацию реалистичной,
            # сохраняя результат одинаковым при каждом запуске.
            variation = ((day_offset + phase) % 5) - 1
            quantity = max(0, round((base + variation) * weekday_factor))
            rows.append({"Дата": pd.Timestamp(sale_date), "Товар": product, "Количество": quantity})
    return pd.DataFrame(rows, columns=SALES_COLUMNS)


def normalize_sales(source: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Проверяет колонки и нормализует даты, наименования и количества."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing_columns:
        missing_text = ", ".join(f"«{column}»" for column in missing_columns)
        return pd.DataFrame(columns=SALES_COLUMNS), (
            "Неверный формат CSV: отсутствуют обязательные колонки "
            f"{missing_text}. Ожидаются колонки «Дата», «Товар», «Количество»."
        )

    sales = source[SALES_COLUMNS].copy()
    sales["Дата"] = pd.to_datetime(sales["Дата"], errors="coerce")
    sales["Товар"] = sales["Товар"].fillna("").astype(str).str.strip()
    sales["Количество"] = pd.to_numeric(sales["Количество"], errors="coerce").fillna(0)
    sales = sales.dropna(subset=["Дата"])
    sales = sales[sales["Товар"] != ""].copy()
    sales["Количество"] = sales["Количество"].astype(float)
    return sales, None


def parse_uploaded_csv(uploaded_file: object) -> tuple[pd.DataFrame, str | None]:
    """Читает CSV в UTF-8/Windows-1251 и возвращает нормализованные продажи."""
    raw_bytes = uploaded_file.getvalue()  # type: ignore[attr-defined]
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            source = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=None, engine="python")
            return normalize_sales(source)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            last_error = error
    return pd.DataFrame(columns=SALES_COLUMNS), (
        "Не удалось прочитать CSV. Проверьте разделитель и кодировку файла "
        f"(UTF-8 или Windows-1251). Детали: {last_error}"
    )


def initialize_inventory(sales: pd.DataFrame, default_lead_time: int, fingerprint: str) -> None:
    """Создаёт остатки по SKU один раз для каждого нового набора данных."""
    if st.session_state.get("inventory_fingerprint") == fingerprint:
        return
    products = sorted(sales["Товар"].dropna().astype(str).unique().tolist())
    inventory = pd.DataFrame(
        [{"Товар": product, "Остаток": 0, "Срок поставки": default_lead_time} for product in products],
        columns=INVENTORY_COLUMNS,
    )
    st.session_state["inventory_table"] = inventory
    st.session_state["inventory_fingerprint"] = fingerprint


def calculate_results(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    history_days: int,
    horizon_days: int,
    default_lead_time: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    """Рассчитывает спрос и заказ по формулам из технического задания."""
    if sales.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    maximum_date = sales["Дата"].max().normalize()
    window_start = maximum_date - pd.Timedelta(days=history_days - 1)
    period_sales = sales[(sales["Дата"] >= window_start) & (sales["Дата"] < maximum_date + pd.Timedelta(days=1))]

    stock = inventory.copy() if not inventory.empty else pd.DataFrame(columns=INVENTORY_COLUMNS)
    stock["Товар"] = stock["Товар"].astype(str)
    stock["Остаток"] = pd.to_numeric(stock["Остаток"], errors="coerce").fillna(0)
    stock["Срок поставки"] = pd.to_numeric(stock["Срок поставки"], errors="coerce")
    stock_by_product = stock.drop_duplicates("Товар", keep="last").set_index("Товар")

    totals = period_sales.groupby("Товар", as_index=False)["Количество"].sum()
    all_products = sorted(set(sales["Товар"].astype(str)) | set(stock["Товар"]))
    result = pd.DataFrame({"Товар": all_products}).merge(totals, on="Товар", how="left")
    result["Количество"] = result["Количество"].fillna(0)
    result["Остаток"] = result["Товар"].map(stock_by_product["Остаток"]).fillna(0)
    result["Срок поставки"] = result["Товар"].map(stock_by_product["Срок поставки"]).fillna(default_lead_time)
    result["Средний спрос в день"] = (result["Количество"] / history_days).round(2)
    result["Прогноз"] = (result["Средний спрос в день"] * horizon_days).round(0).astype(int)
    result["К заказу"] = result.apply(
        lambda row: math.ceil(max(0, row["Средний спрос в день"] * row["Срок поставки"] - row["Остаток"])),
        axis=1,
    )
    result = result.rename(columns={"Количество": "Продано за период"})
    result["Продано за период"] = result["Продано за период"].round(2)
    result = result.sort_values(["К заказу", "Товар"], ascending=[False, True]).reset_index(drop=True)
    return result, period_sales, maximum_date


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">DEMAND PLANNING · MVP</div>
          <h1>Demand Studio</h1>
          <p>Планируйте закупки на основе истории продаж и текущих остатков.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: radial-gradient(ellipse at 15% 0%, #172540 0, #0b1020 42%, #090d17 100%); }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #111a2d, #0c1220); border-right: 1px solid #24314b; }
      .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero { padding: 1.45rem 1.65rem; border: 1px solid rgba(143, 169, 219, .2); border-radius: 22px;
        background: linear-gradient(115deg, rgba(31, 48, 79, .92), rgba(17, 26, 44, .78)); margin-bottom: 1.35rem; }
      .hero h1 { margin: .25rem 0 .35rem; font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -.045em; color: #f3f7ff; }
      .hero p { margin: 0; color: #aab8d3; font-size: 1rem; }
      .eyebrow { color: #8eaef8; font-size: .72rem; font-weight: 700; letter-spacing: .16em; }
      [data-testid="stMetric"] { min-height: 112px; padding: 1rem 1.1rem; border-radius: 16px;
        border: 1px solid rgba(143, 169, 219, .18); background: linear-gradient(145deg, #151f33, #101827); }
      [data-testid="stMetricLabel"] { color: #9eacc7; }
      [data-testid="stMetricValue"] { color: #f2f6ff; }
      div[data-testid="stTabs"] button { font-weight: 600; }
      div[data-testid="stDataFrame"] { border: 1px solid #26344d; border-radius: 14px; overflow: hidden; }
      div[data-testid="stFileUploader"] { border: 1px dashed #3b4e70; border-radius: 16px; padding: .7rem; background: rgba(18, 28, 46, .5); }
      @media (max-width: 700px) { .block-container { padding: 1rem .8rem 2rem; } .hero { padding: 1.15rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.markdown("## ⚙️ Параметры расчёта")
history_days = st.sidebar.number_input("Окно истории, дней", min_value=1, max_value=3650, value=28, step=1)
horizon_days = st.sidebar.number_input("Горизонт планирования, дней", min_value=1, max_value=365, value=7, step=1)
default_lead_time = st.sidebar.number_input("Срок поставки по умолчанию, дней", min_value=0, max_value=365, value=5, step=1)
st.sidebar.caption("Срок поставки по умолчанию используется для новых SKU. Значение можно изменить в таблице остатков.")

render_header()
upload_tab, results_tab, guide_tab = st.tabs(
    ["📥 Загрузка данных и Остатки", "📊 Результаты и Аналитика", "📚 Инструкция"]
)

with upload_tab:
    st.subheader("Загрузите историю продаж")
    st.markdown("CSV должен содержать колонки **Дата**, **Товар**, **Количество**.")
    uploaded_file = st.file_uploader("Файл продаж (CSV)", type=["csv"], key="sales_file_uploader")

    if uploaded_file is None:
        source_sales = create_demo_sales()
        source_fingerprint = "demo-data-v1"
        is_demo = True
        st.info("Показываются демонстрационные данные. Загрузите свой CSV-файл на вкладке 1")
    else:
        source_fingerprint = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        source_sales, parse_error = parse_uploaded_csv(uploaded_file)
        is_demo = False
        if parse_error:
            st.error(parse_error)
            st.session_state["sales_data"] = pd.DataFrame(columns=SALES_COLUMNS)
            st.session_state["inventory_table"] = pd.DataFrame(columns=INVENTORY_COLUMNS)
            st.session_state["is_demo_data"] = False
        elif source_sales.empty:
            st.error("В CSV не осталось строк с корректными датами и названиями товаров. Проверьте данные и загрузите файл повторно.")
            st.session_state["sales_data"] = pd.DataFrame(columns=SALES_COLUMNS)
            st.session_state["inventory_table"] = pd.DataFrame(columns=INVENTORY_COLUMNS)
            st.session_state["is_demo_data"] = False
        else:
            st.success(f"Файл загружен: {source_sales['Товар'].nunique()} SKU, {len(source_sales):,} строк продаж.")

    can_calculate = not source_sales.empty and (uploaded_file is None or "parse_error" not in locals() or parse_error is None)
    if can_calculate:
        initialize_inventory(source_sales, int(default_lead_time), source_fingerprint)
        st.markdown("#### Остатки и сроки поставки по SKU")
        st.caption("Изменения применяются сразу и сохраняются при переключении между вкладками.")
        edited_inventory = st.data_editor(
            st.session_state["inventory_table"],
            key=f"inventory_editor_{source_fingerprint[:12]}",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Товар": st.column_config.TextColumn("Товар / SKU", disabled=True, width="large"),
                "Остаток": st.column_config.NumberColumn("Остаток", min_value=0, step=1, format="%d", required=True),
                "Срок поставки": st.column_config.NumberColumn("Срок поставки, дней", min_value=0, step=1, format="%d", required=True),
            },
        )
        st.session_state["inventory_table"] = edited_inventory.copy()
        st.session_state["sales_data"] = source_sales.copy()
        st.session_state["source_fingerprint"] = source_fingerprint
        st.session_state["is_demo_data"] = is_demo
        st.session_state["calculation_settings"] = {
            "history_days": int(history_days),
            "horizon_days": int(horizon_days),
            "default_lead_time": int(default_lead_time),
        }
    elif uploaded_file is None:
        # Защитное состояние: демонстрационный набор всегда должен дать результат.
        st.session_state["sales_data"] = source_sales.copy()
        st.session_state["is_demo_data"] = True

with results_tab:
    sales_data = st.session_state.get("sales_data", create_demo_sales())
    inventory_data = st.session_state.get("inventory_table", pd.DataFrame(columns=INVENTORY_COLUMNS))
    settings = st.session_state.get(
        "calculation_settings",
        {"history_days": int(history_days), "horizon_days": int(horizon_days), "default_lead_time": int(default_lead_time)},
    )
    if st.session_state.get("is_demo_data", True):
        st.info("Показываются демонстрационные данные. Загрузите свой CSV-файл на вкладке 1")
    if not sales_data.empty:
        results, period_sales, maximum_date = calculate_results(
            sales_data,
            inventory_data,
            settings["history_days"],
            settings["horizon_days"],
            settings["default_lead_time"],
        )
        if results.empty:
            st.warning("Нет данных для расчёта за выбранное окно истории.")
        else:
            st.session_state["calculation_results"] = results.copy()
            top_product = results.sort_values(["Продано за период", "Товар"], ascending=[False, True]).iloc[0]
            ordered_count = int((results["К заказу"] > 0).sum())
            metric_columns = st.columns(3)
            metric_columns[0].metric("Всего SKU", f"{len(results):,}")
            metric_columns[1].metric("Товаров к заказу", f"{ordered_count:,}")
            metric_columns[2].metric("Топ-товар", str(top_product["Товар"]), f"{top_product['Продано за период']:,.0f} шт. за период")

            st.markdown("#### Продажи и прогноз по товарам")
            chart_data = results.set_index("Товар")[["Продано за период", "Прогноз"]]
            st.bar_chart(chart_data, color=["#547fe8", "#58c6ae"], height=270)
            st.caption(
                f"Окно истории: {settings['history_days']} календарных дней, "
                f"до {maximum_date.strftime('%d.%m.%Y')} включительно. "
                f"Горизонт прогноза: {settings['horizon_days']} дней."
            )

            st.markdown("#### Результаты расчёта")
            search_query = st.text_input("Поиск по товару / SKU", placeholder="Например, Nike или SKU-104")
            only_to_order = st.checkbox("Показать только товары, требующие закупки")
            filtered_results = results.copy()
            if search_query.strip():
                filtered_results = filtered_results[
                    filtered_results["Товар"].str.contains(search_query.strip(), case=False, regex=False, na=False)
                ]
            if only_to_order:
                filtered_results = filtered_results[filtered_results["К заказу"] > 0]
            display_columns = [
                "Товар",
                "Продано за период",
                "Средний спрос в день",
                "Прогноз",
                "Остаток",
                "К заказу",
            ]
            st.dataframe(
                filtered_results[display_columns],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Товар": st.column_config.TextColumn("Товар", width="large"),
                    "Продано за период": st.column_config.NumberColumn("Продано за период", format="%.2f"),
                    "Средний спрос в день": st.column_config.NumberColumn("Средний спрос в день", format="%.2f"),
                    "Прогноз": st.column_config.NumberColumn("Прогноз", format="%d"),
                    "Остаток": st.column_config.NumberColumn("Остаток", format="%d"),
                    "К заказу": st.column_config.NumberColumn("К заказу", format="%d"),
                },
            )
            csv_data = filtered_results[display_columns].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Скачать результат CSV",
                data=csv_data,
                file_name="prognoz_sprosa.csv",
                mime="text/csv",
                use_container_width=False,
            )
    else:
        st.warning("Загрузите CSV с колонками «Дата», «Товар», «Количество» на вкладке 1, чтобы рассчитать результаты.")

with guide_tab:
    st.subheader("Как пользоваться Demand Studio")
    st.markdown(
        """
        1. **Настройте расчёт** в боковой панели: окно истории, горизонт планирования и срок поставки по умолчанию.
        2. **Загрузите CSV** на вкладке «Загрузка данных и Остатки». Нужны три колонки: `Дата`, `Товар`, `Количество`.
        3. **Заполните остатки** и при необходимости сроки поставки в таблице по каждому SKU. Данные сохраняются между вкладками.
        4. **Изучите прогноз** на вкладке «Результаты и Аналитика»: используйте поиск и фильтр закупки, затем скачайте текущую выборку в CSV.

        Даты, которые не удалось распознать, исключаются. Некорректное или пустое количество считается нулём.
        Средний спрос рассчитывается за указанное число календарных дней относительно самой поздней даты в файле.
        Рекомендация к заказу округляется вверх до целого количества.
        """
    )
