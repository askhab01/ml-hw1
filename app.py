import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# пути считаем от папки, где лежит app.py, тогда приложение работает и локально, и в облаке
FOLDER = Path(__file__).parent
MODEL_PATH = FOLDER / "model.pickle"
DATA_PATH = FOLDER / "cars_train_clean.csv"


@st.cache_resource  # модель читаем с диска один раз, а не при каждом клике
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data  # таблицу для графиков тоже читаем один раз
def load_data():
    return pd.read_csv(DATA_PATH)


def show(fig):
    """Выводит график на страницу и закрывает его, чтобы графики не копились в памяти сервера"""
    st.pyplot(fig)
    plt.close(fig)


def money(value):
    """Число с пробелами между тысячами, 1234567 превращается в 1 234 567"""
    return f"{value:,.0f}".replace(",", " ")


def prepare(df, medians):
    """Повторяет предобработку из ноутбука, чтобы модель получила данные в привычном виде"""
    df = df.copy()
    # в исходных файлах значения записаны с единицами измерения, например '23.4 kmpl'
    for col in ["mileage", "engine", "max_power"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.split().str[0], errors="coerce")
    # нулевой расход и нулевая мощность считаем пропусками, как в ноутбуке
    df[["mileage", "max_power"]] = df[["mileage", "max_power"]].replace(0, np.nan)
    # пропуски заполняем медианами, посчитанными на обучающей выборке
    for col, value in medians.items():
        df[col] = df[col].fillna(value)
    df[["engine", "seats"]] = df[["engine", "seats"]].astype(int)
    return df


if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Не хватает файлов model.pickle или cars_train_clean.csv. Они создаются при запуске ноутбука.")
    st.stop()

artifacts = load_artifacts()
model = artifacts["model"]
real_features = artifacts["real_features"]
cat_features = artifacts["cat_features"]
features = real_features + cat_features
categories = artifacts["categories"]
medians = artifacts["medians"]
data = load_data()

st.title("Цена подержанного автомобиля")
st.caption("Ridge-регрессия на характеристиках машины. Цены в валюте исходного датасета.")

page = st.sidebar.selectbox("Раздел", ["Данные", "Прогноз", "Веса модели"])

st.sidebar.metric("R² на тесте", f"{artifacts['metrics']['R2 test']:.3f}")
st.sidebar.metric("Прогнозов в пределах 10% от цены", f"{artifacts['metrics']['business test']:.1%}")


# ------------------------------ Данные ------------------------------
if page == "Данные":
    st.header("На каких данных училась модель")

    left, middle, right = st.columns(3)
    left.metric("Машин", money(len(data)))
    middle.metric("Медианная цена", money(data["selling_price"].median()))
    right.metric("Годы выпуска", f"{data['year'].min()}-{data['year'].max()}")

    log_price = st.checkbox("Логарифмическая шкала цены", value=True)
    st.caption("У цены длинный хвост из дорогих машин. В обычной шкале они сжимают графики, в логарифмической видно всех.")

    st.subheader("Распределение цены")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    # log_scale строит столбцы одинаковой ширины уже в логарифмической шкале
    sns.histplot(data["selling_price"], bins=50, log_scale=log_price, ax=ax)
    ax.set_xlabel("Цена")
    ax.set_ylabel("Машин")
    show(fig)

    st.subheader("Цена и числовой признак")
    num_col = st.selectbox("Признак", real_features, index=real_features.index("max_power"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(data[num_col], data["selling_price"], s=8, alpha=0.4)
    if log_price:
        ax.set_yscale("log")
    ax.set_xlabel(num_col)
    ax.set_ylabel("Цена")
    show(fig)

    st.subheader("Цена по категориям")
    cat_col = st.selectbox("Категория", cat_features)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=data, x=cat_col, y="selling_price", ax=ax)
    if log_price:
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Цена")
    ax.tick_params(axis="x", rotation=20)
    show(fig)

    st.subheader("Корреляции числовых признаков")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = data[real_features + ["seats", "selling_price"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    show(fig)


# ------------------------------ Прогноз ------------------------------
if page == "Прогноз":
    st.header("Прогноз для одной машины")

    # форма нужна, чтобы прогноз считался по кнопке, а не после каждого изменения поля
    with st.form("car"):
        left, right = st.columns(2)
        with left:
            year = st.number_input("Год выпуска", min_value=1980, max_value=2025, value=2015)
            km_driven = st.number_input("Пробег, км", min_value=0, max_value=3_000_000, value=70_000, step=5_000)
            mileage = st.number_input("Расход, км на литр", min_value=5.0, max_value=45.0, value=19.4, step=0.5)
            engine = st.number_input("Объём двигателя, куб. см", min_value=600, max_value=4000, value=1248, step=50)
            max_power = st.number_input("Мощность, л.с.", min_value=30.0, max_value=400.0, value=82.0, step=1.0)
        with right:
            # варианты в списках взяты из обучающей выборки, других значений модель не знает
            fuel = st.selectbox("Топливо", categories["fuel"], index=categories["fuel"].index("Petrol"))
            seller_type = st.selectbox(
                "Продавец", categories["seller_type"], index=categories["seller_type"].index("Individual")
            )
            transmission = st.selectbox(
                "Коробка передач", categories["transmission"], index=categories["transmission"].index("Manual")
            )
            owner = st.selectbox("Владелец", categories["owner"])
            seats = st.selectbox("Число мест", categories["seats"], index=categories["seats"].index(5))
        submitted = st.form_submit_button("Предсказать")

    if submitted:
        # модель ждёт таблицу, поэтому одна машина это таблица из одной строки
        car = pd.DataFrame([{
            "year": year,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "seats": seats,
        }])
        price = float(model.predict(car)[0])
        st.metric("Прогноз цены", money(price))
        if price <= 0:
            st.warning(
                "Прогноз получился отрицательным. Линейная модель не знает, что цена не бывает меньше нуля, "
                "и для старых и слабых машин может уйти в минус. Такому прогнозу верить нельзя."
            )

    st.header("Прогноз по файлу")
    uploaded = st.file_uploader("CSV с машинами", type="csv")  # пока файла нет, здесь None
    if uploaded is None:
        st.info("Загрузите CSV со столбцами " + ", ".join(features) + ". Подойдёт файл sample_cars.csv из репозитория.")
    else:
        batch = pd.read_csv(uploaded)
        missing = [col for col in features if col not in batch.columns]
        if missing:
            st.error("В файле нет столбцов " + ", ".join(missing))
        else:
            batch = prepare(batch, medians)
            if batch[features].isna().any().any():
                st.error("В столбцах year, km_driven или в категориальных столбцах есть пустые значения, заполните их.")
            else:
                # незнакомые категории модель не умеет учитывать и молча считает их базовыми, об этом стоит сказать
                for col in cat_features:
                    unknown = sorted(set(batch[col]) - set(categories[col]), key=str)
                    if unknown:
                        st.warning(f"В столбце {col} есть значения, которых не было при обучении. Это {unknown}.")
                # лишние столбцы в файле не мешают, в модель идут только нужные
                batch["predicted_price"] = model.predict(batch[features]).round()
                n_negative = int((batch["predicted_price"] <= 0).sum())
                st.metric("Машин в файле", len(batch))
                if n_negative:
                    st.warning(f"Отрицательных прогнозов {n_negative}. Для этих машин модель не подходит.")
                st.dataframe(batch, hide_index=True)
                st.download_button(
                    "Скачать прогнозы",
                    data=batch.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    on_click="ignore",  # скачивание не перезапускает страницу
                )


# ------------------------------ Веса модели ------------------------------
if page == "Веса модели":
    st.header("Веса обученной модели")

    # названия признаков после кодирования выглядят как num__year и cat__fuel_Diesel, приставку убираем
    names = [name.split("__")[1] for name in model.named_steps["preprocessor"].get_feature_names_out()]
    coefs = pd.Series(model.named_steps["model"].coef_, index=names)
    coefs = coefs.reindex(coefs.abs().sort_values().index)  # самые большие по модулю окажутся вверху графика

    st.metric("Свободный член", money(model.named_steps["model"].intercept_))

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["tab:red" if value < 0 else "tab:green" for value in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Вес")
    show(fig)

    st.markdown(
        "**Как читать.** Зелёные признаки увеличивают прогноз цены, красные уменьшают.\n\n"
        "Числовые признаки стандартизированы, поэтому их вес показывает, на сколько меняется цена, "
        "когда признак вырастает на одно стандартное отклонение.\n\n"
        "У категориальных признаков вес это разница с базовой категорией. "
        "Базовые категории CNG, Dealer, Automatic, First Owner и 2 места."
    )

    # та же информация таблицей, от самого сильного признака к самому слабому
    table = coefs.sort_values(key=abs, ascending=False).round()
    st.dataframe(pd.DataFrame({"признак": table.index, "вес": table.values}), hide_index=True)
