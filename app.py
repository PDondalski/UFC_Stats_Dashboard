import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ============= Ustawienie zachowania strony dashboardu =============
st.set_page_config(layout="wide", page_title="Wyniki walk UFC")


# ============= Załadowanie i przygotowanie danych =============
@st.cache_data
def load_data():
    # Załadowanie zbioru danych do ramki danych
    df = pd.read_csv("data/ufc_event_fight_stats.csv", delimiter=",")

    # Wybranie kolumn, które będą potrzebne do analizy
    df = df[[
        "f1_id",
        "f2_id",
        "f1_name",
        "f2_name",
        "weight_class",
        "f1_age_during",
        "f2_age_during",
        "f1_sig_strikes",
        "f2_sig_strikes",
        "f1_takedown_atts",
        "f2_takedown_atts",
        "f1_takedowns",
        "f2_takedowns",
        "result"
    ]]

    # Wybór głównych kategorii wagowych
    valid_weights = [
        "Flyweight Bout",
        "Bantamweight Bout",
        "Featherweight Bout",
        "Lightweight Bout",
        "Welterweight Bout",
        "Middleweight Bout",
        "Light Heavyweight Bout",
        "Heavyweight Bout"
    ]

    df = df[df["weight_class"].isin(valid_weights)].copy()

    # Usunięcie braków danych w kolumnach potrzebnych do dalszej analizy
    df = df.dropna(subset=[
        "f1_age_during",
        "f2_age_during",
        "f1_sig_strikes",
        "f2_sig_strikes",
        "f1_takedown_atts",
        "f2_takedown_atts",
        "f1_takedowns",
        "f2_takedowns"
    ])

    # Ustalenie, który zawodnik wygrał daną walkę
    def get_winner_side(row):
        if str(row["result"]) == str(row["f1_id"]):
            return "f1"
        elif str(row["result"]) == str(row["f2_id"]):
            return "f2"
        else:
            return "brak"

    df["winner_side"] = df.apply(get_winner_side, axis=1)

    # Zostawienie tylko walk zakończonych zwycięstwem jednego z zawodników
    df = df[df["winner_side"] != "brak"].copy()

    # Przygotowanie kolumn opisujących zwycięzcę i przegranego
    winner_is_f1 = df["winner_side"] == "f1"

    df["winner_name"] = np.where(winner_is_f1, df["f1_name"], df["f2_name"])
    df["loser_name"] = np.where(winner_is_f1, df["f2_name"], df["f1_name"])

    df["winner_age"] = np.where(winner_is_f1, df["f1_age_during"], df["f2_age_during"])
    df["loser_age"] = np.where(winner_is_f1, df["f2_age_during"], df["f1_age_during"])

    df["winner_sig_strikes"] = np.where(winner_is_f1, df["f1_sig_strikes"], df["f2_sig_strikes"])
    df["loser_sig_strikes"] = np.where(winner_is_f1, df["f2_sig_strikes"], df["f1_sig_strikes"])

    df["winner_takedown_atts"] = np.where(winner_is_f1, df["f1_takedown_atts"], df["f2_takedown_atts"])
    df["loser_takedown_atts"] = np.where(winner_is_f1, df["f2_takedown_atts"], df["f1_takedown_atts"])

    df["winner_takedowns"] = np.where(winner_is_f1, df["f1_takedowns"], df["f2_takedowns"])
    df["loser_takedowns"] = np.where(winner_is_f1, df["f2_takedowns"], df["f1_takedowns"])

    # Przygotowanie kolumn do analizy wieku
    df["age_diff"] = df["winner_age"] - df["loser_age"]
    df["age_diff_abs"] = abs(df["age_diff"])
    df["age_diff_group"] = pd.cut(
        df["age_diff_abs"],
        bins=[-1, 0, 2, 5, 10, 100],
        labels=["0 lat", "1-2 lata", "3-5 lat", "6-10 lat", "11+ lat"]
    )
    df["age_result"] = df["age_diff"].apply(
        lambda value: "Wygrał starszy" if value > 0 else ("Wygrał młodszy" if value < 0 else "Ten sam wiek")
    )

    # Przygotowanie kolumn do porównania przewag w walce
    df["striking_result"] = df.apply(
        lambda row: "Zwycięzca miał więcej uderzeń" if row["winner_sig_strikes"] > row["loser_sig_strikes"]
        else ("Przegrany miał więcej uderzeń" if row["winner_sig_strikes"] < row["loser_sig_strikes"] else "Tyle samo uderzeń"),
        axis=1
    )

    df["takedown_result"] = df.apply(
        lambda row: "Zwycięzca miał więcej obaleń" if row["winner_takedowns"] > row["loser_takedowns"]
        else ("Przegrany miał więcej obaleń" if row["winner_takedowns"] < row["loser_takedowns"] else "Tyle samo obaleń"),
        axis=1
    )

    return df


# Załadowanie danych
df = load_data()


# ============= Dashboard =============
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Przejdź do:", ["Wprowadzenie", "Eksploracja danych", "Model", "Wnioski"])


# >>>>>>> 1. WPROWADZENIE <<<<<<<
if page == "Wprowadzenie":
    st.title("Co odróżnia zwycięzców od przegranych w walkach UFC?")

    st.markdown("""
    Ten dashboard prezentuje wyniki walk mieszanych sztuk walki (MMA) stoczonych w federacji UFC na podstawie danych z 
    Kaggle, pobranych z oficjalnej strony UFCStats w listopadzie 2024 roku. Celem raportu jest sprawdzenie,
    które elementy statystyczne najczęściej towarzyszą zwycięstwu. W ramach tej analizy skupiono się na 
    wieku zawodników, liczbie znaczących uderzeń oraz liczbie prób i skutecznych obaleń.

    Analiza skupia się na porównaniu **zwycięzców z przegranymi** oraz **starszych zawodników z młodszymi**.

    Dane zostały odpowiednio przygotowane do analizy: raport skupia się na głównych kategoriach
    wagowych i pomija walki bez jednoznacznego zwycięzcy, aby dalsze porównania mogły opierać się
    na parach **zwycięzca - przegrany**.

    W ramach raportu:
    - Przedstawiona zostanie liczba walk w wybranych, głównych kategoriach wagowych
    - Sprawdzona zostanie zależność między wiekiem zawodników a wynikiem walki
    - Porównana zostanie aktywność zwycięzców i przegranych zarówno w stójce, jak i w parterze
    - Zaprezentowany zostanie model regresyjny do przewidywania liczby skutecznych obaleń

    ---
    Dane pochodzą ze zbioru 
    [UFC Dataset - Kaggle](https://www.kaggle.com/datasets/thasankakandage/ufc-dataset-2024?resource=download) 
    i zostały udostępnione na licencji Apache 2.0.
    """)


# >>>>>>> 2. EKSPLORACJA <<<<<<<
elif page == "Eksploracja danych":
    st.title("Eksploracja danych")

    st.sidebar.header("Filtry danych")

    # Filtry pozwalają użytkownikowi zawęzić analizę do wybranych kategorii wagowych i różnicy wieku
    selected_weights = st.sidebar.multiselect(
        "Wybierz kategorie wagowe",
        options=sorted(df["weight_class"].unique()),
        default=sorted(df["weight_class"].unique())
    )

    min_age_diff = int(df["age_diff_abs"].min())
    max_age_diff = int(df["age_diff_abs"].max())

    age_diff_range = st.sidebar.slider(
        "Różnica wieku między zawodnikami",
        min_value=min_age_diff,
        max_value=max_age_diff,
        value=(min_age_diff, max_age_diff)
    )

    filtered_df = df[
        (df["weight_class"].isin(selected_weights)) &
        (df["age_diff_abs"].between(*age_diff_range))
    ].copy()

    if filtered_df.empty:
        st.warning("Brak walk spełniających wybrane kryteria filtrowania.")
        st.stop()

    st.markdown("""
    W tej części raportu można wybrać kategorie wagowe oraz zakres różnicy wieku między zawodnikami.
    Na podstawie tych filtrów aktualizowane są wykresy i tabela, dzięki czemu użytkownik może samodzielnie
    sprawdzać zależności widoczne w danych.

    Eksploracja zaczyna się od ogólnej liczby walk w kategoriach wagowych, a następnie przechodzi do porównań
    zwycięzców i przegranych pod względem wieku, znaczących uderzeń oraz obaleń.
    """)

    # Krótkie podsumowanie danych po filtrowaniu
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liczba walk", len(filtered_df))
    col2.metric("Kategorie wagowe", filtered_df["weight_class"].nunique())
    col3.metric("Średni wiek zwycięzcy", int(round(filtered_df['winner_age'].mean())))
    col4.metric("Średnia różnica wieku", int(round(filtered_df['age_diff_abs'].mean())))

    st.subheader("Ogólny obraz danych")

    # +++++++++++ Wykres - liczba walk w danej kategorii wagowej +++++++++++
    # Wykres pokazuje, które kategorie wagowe mają największy udział w analizowanych danych
    weight_counts = filtered_df["weight_class"].value_counts().reset_index()
    weight_counts.columns = ["weight_class", "fight_count"]

    fig = px.bar(
        weight_counts,
        x="weight_class",
        y="fight_count",
        title="Liczba walk w poszczególnych kategoriach wagowych",
        labels={"weight_class": "Kategoria wagowa", "fight_count": "Liczba walk"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Czy wiek ma znaczenie?")

    # +++++++++++ Wykres - czy częściej wygrywa zawodnik starszy czy młodszy +++++++++++
    # Wykres pozwala sprawdzić, czy częściej wygrywają zawodnicy starsi czy młodsi
    fig = px.histogram(
        filtered_df,
        x="age_result",
        color="age_result",
        title="Czy częściej wygrywa zawodnik starszy czy młodszy?",
        labels={"age_result": "Relacja wieku", "count": "Liczba walk"},
        category_orders={"age_result": ["Wygrał młodszy", "Ten sam wiek", "Wygrał starszy"]}
    )
    st.plotly_chart(fig, use_container_width=True)

    # +++++++++++ Wykres - różnica wieku w jednej walce +++++++++++
    # Wykres pokazuje, jak duże różnice wieku najczęściej występują w walkach
    age_diff_counts = filtered_df["age_diff_group"].value_counts().sort_index().reset_index()
    age_diff_counts.columns = ["age_diff_group", "fight_count"]

    fig = px.bar(
        age_diff_counts,
        x="age_diff_group",
        y="fight_count",
        title="Jak bardzo wiekiem różnią się zawodnicy w jednej walce?",
        labels={"age_diff_group": "Różnica wieku", "fight_count": "Liczba walk"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stójka: znaczące uderzenia")

    # +++++++++++ Wykres - porównanie ilości znaczących uderzeń obydwu zawodników +++++++++++
    # Wykres porównuje aktywność zwycięzcy i przegranego w stójce
    fig = px.scatter(
        filtered_df,
        x="winner_sig_strikes",
        y="loser_sig_strikes",
        color="weight_class",
        hover_data=["winner_name", "loser_name", "age_result"],
        title="Znaczące uderzenia zwycięzcy i przegranego",
        labels={
            "winner_sig_strikes": "Znaczące uderzenia zwycięzcy",
            "loser_sig_strikes": "Znaczące uderzenia przegranego",
            "weight_class": "Kategoria wagowa"
        },
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    # +++++++++++ Wykres - pokazanie ilości znaczących uderzeń u zwycięzcy i przegranego +++++++++++
    # Wykres pokazuje, jak często przewaga w uderzeniach pokrywa się ze zwycięstwem
    fig = px.histogram(
        filtered_df,
        x="striking_result",
        color="striking_result",
        title="Czy zwycięzca częściej zadaje więcej znaczących uderzeń?",
        labels={"striking_result": "Wynik porównania", "count": "Liczba walk"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Parter: próby i skuteczne obalenia")

    # +++++++++++ Wykres - związek między próbami, a skutecznymi obeleniami +++++++++++
    # Wykres sprawdza związek między próbami obaleń a skutecznymi obaleniami zwycięzcy
    fig = px.scatter(
        filtered_df,
        x="winner_takedown_atts",
        y="winner_takedowns",
        color="weight_class",
        hover_data=["winner_name", "loser_name"],
        title="Próby obaleń a skuteczne obalenia zwycięzców",
        labels={
            "winner_takedown_atts": "Próby obaleń zwycięzcy",
            "winner_takedowns": "Skuteczne obalenia zwycięzcy",
            "weight_class": "Kategoria wagowa"
        },
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela przedstawiające dane po zastosowaniu filtrów dla dokładnej analizy wyników
    st.subheader("Dane po zastosowaniu filtrów")
    st.dataframe(filtered_df[[
        "winner_name",
        "loser_name",
        "weight_class",
        "winner_age",
        "loser_age",
        "age_result",
        "winner_sig_strikes",
        "loser_sig_strikes",
        "winner_takedowns",
        "loser_takedowns"
    ]])


# >>>>>>> 3. MODEL <<<<<<<
elif page == "Model":
    st.title("Model predykcyjny")

    st.markdown("""
    W tej sekcji użytkownik może sprawdzić przewidywaną liczbę skutecznych obaleń zawodnika
    na podstawie wieku, kategorii wagowej oraz liczby prób obaleń. Model nie przewiduje zwycięzcy
    walki, tylko pokazuje zależność między między liczbą prób obaleń a liczbą skutecznych obaleń.
    """)

    # Przygotowanie danych - zawodnik 1
    f1_model_df = df[["weight_class", "f1_age_during", "f1_takedown_atts", "f1_takedowns"]].rename(columns={
        "f1_age_during": "age",
        "f1_takedown_atts": "takedown_atts",
        "f1_takedowns": "takedowns"
    })

    # Przygotowanie danych - zawodnik 2
    f2_model_df = df[["weight_class", "f2_age_during", "f2_takedown_atts", "f2_takedowns"]].rename(columns={
        "f2_age_during": "age",
        "f2_takedown_atts": "takedown_atts",
        "f2_takedowns": "takedowns"
    })

    # Połączenie danych zawodnika 1 i zawodnika 2 w jeden zbiór do modelu
    model_df = pd.concat([f1_model_df, f2_model_df], ignore_index=True).dropna()

    # One-hot encoding dla kategorii wagowej
    model_df_encoded = pd.get_dummies(model_df, columns=["weight_class"])

    # Cechy wejściowe (X) i zmienna przewidywana (y)
    X = model_df_encoded.drop("takedowns", axis=1)
    y = model_df_encoded["takedowns"]

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Utworzenie i wytrenowanie modelu regresji liniowej
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predykcja dla zbioru testowego
    y_pred = model.predict(X_test)

    # Obliczenie metryk jakości modelu
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Wyświetlenie metryk jakości modelu
    st.subheader("Jakość modelu")
    col1, col2 = st.columns(2)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")

    # Formularz pozwalający użytkownikowi samodzielne przetestowanie działania modelu
    st.subheader("Wprowadź dane do predykcji")

    with st.form("prediction_form"):
        weight_class_choice = st.selectbox(
            "Wybierz kategorię wagową",
            sorted(model_df["weight_class"].unique())
        )

        age_input = st.number_input(
            "Podaj wiek zawodnika",
            min_value=18,
            max_value=60,
            value=30,
            step=1
        )

        takedown_atts_input = st.number_input(
            "Podaj liczbę prób obaleń",
            min_value=0,
            max_value=30,
            value=5,
            step=1
        )

        submitted = st.form_submit_button("Oblicz predykcję")

    # Obsługa kliknięcia przycisku
    if submitted:
        input_df = pd.DataFrame({
            "age": [age_input],
            "takedown_atts": [takedown_atts_input],
            "weight_class": [weight_class_choice]
        })

        # One-hot encoding dla kategorii wagowej
        input_encoded = pd.get_dummies(input_df, columns=["weight_class"])

        # Ustawienie takich samych kolumn jak w danych treningowych
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        # Predykcja i zaokrąglenie wyniku do liczby całkowitej
        prediction = model.predict(input_encoded)[0]
        prediction_rounded = max(0, int(round(prediction)))

        # Wyświetlenie wyniku w dashboardzie
        st.success(
            f"Przewidywana liczba skutecznych obaleń: {prediction_rounded}"
        )


# >>>>>>> 4. WNIOSKI <<<<<<<
elif page == "Wnioski":
    st.title("Wnioski")

    # Przygotowanie prostych wartości liczbowych do podsumowania raportu
    fight_count = len(df)
    younger_wins = df["age_result"].value_counts().get("Wygrał młodszy", 0)
    older_wins = df["age_result"].value_counts().get("Wygrał starszy", 0)
    winner_more_strikes = df["striking_result"].value_counts().get("Zwycięzca miał więcej uderzeń", 0)
    loser_more_strikes = df["striking_result"].value_counts().get("Przegrany miał więcej uderzeń", 0)
    same_strikes = df["striking_result"].value_counts().get("Tyle samo uderzeń", 0)
    winner_more_takedowns = df["takedown_result"].value_counts().get("Zwycięzca miał więcej obaleń", 0)
    loser_more_takedowns = df["takedown_result"].value_counts().get("Przegrany miał więcej obaleń", 0)
    same_takedowns = df["takedown_result"].value_counts().get("Tyle samo obaleń", 0)

    younger_wins_percent = younger_wins / fight_count * 100
    older_wins_percent = older_wins / fight_count * 100
    winner_more_strikes_percent = winner_more_strikes / fight_count * 100
    loser_more_strikes_percent = loser_more_strikes / fight_count * 100
    same_strikes_percent = same_strikes / fight_count * 100
    winner_more_takedowns_percent = winner_more_takedowns / fight_count * 100
    loser_more_takedowns_percent = loser_more_takedowns / fight_count * 100
    same_takedowns_percent = same_takedowns / fight_count * 100

    st.markdown(f"""
    Na podstawie przygotowanej eksploracji raport prowadzi do kilku najważniejszych wniosków:

    - Z uwagi na fakt, że oznaczenia „zawodnik 1” i „zawodnik 2” nie niosą jasnej informacji analitycznej, 
    dane zostały przekształcone do układu zwycięzca-przegrany, co ułatwiło interpretację wyników
    - W przygotowanym zbiorze analizowanych jest **{fight_count}** walk z jednoznacznym zwycięzcą
    - Młodsi zawodnicy wygrywali częściej niż starsi: **{younger_wins}** walk wygrali młodsi ({younger_wins_percent:.1f}%)
    wobec **{older_wins}** walk, które wygrali starsi zawodnicy ({older_wins_percent:.1f}%)
    - Znaczące uderzenia były mocno powiązane z wynikiem walki: zwycięzca miał ich więcej w **{winner_more_strikes}** walkach 
    ({winner_more_strikes_percent:.1f}%), przegrany w **{loser_more_strikes}** walkach ({loser_more_strikes_percent:.1f}%), 
    a w **{same_strikes}** walkach ({same_strikes_percent:.1f}%) obaj zawodnicy mieli tyle samo znaczących uderzeń
    - Obalenia były mniej jednoznaczne niż znaczące uderzenia: zwycięzca miał ich więcej w **{winner_more_takedowns}** walkach 
    ({winner_more_takedowns_percent:.1f}%), przegrany w **{loser_more_takedowns}** walkach 
    ({loser_more_takedowns_percent:.1f}%), a w **{same_takedowns}** walkach ({same_takedowns_percent:.1f}%) 
    obaj zawodnicy mieli tyle samo skutecznych obaleń
    - Obalenia warto analizować razem z liczbą prób, ponieważ sama liczba skutecznych obaleń nie pokazuje całej 
    aktywności zawodnika
    - Model regresyjny pokazuje, że im więcej prób obaleń, tym większa przewidywana liczba skutecznych obaleń. 
    - Model osiągnął umiarkowaną jakość predykcyjną. 
    Współczynnik R² na poziomie **0.560** wskazuje, że model wyjaśnia ponad połowę zmienności liczby skutecznych obaleń. 
    Z kolei MAE równe **0.692** oznacza, że średni błąd predykcji jest mniejszy niż jedno obalenie, 
    co w kontekście analizowanej zmiennej można uznać za wynik zadowalający.

    Na wynik walki MMA wpływa wiele czynników, takich jak skuteczność w stójce, umiejętność sprowadzenia walki
    do parteru, kontrola pozycji, próby poddań, nokdauny oraz decyzje sędziowskie. Dlatego wyniku pojedynku
    nie należy sprowadzać do jednej statystyki.

    W tym raporcie skupiono się na wybranych elementach, które pozwalają porównać zwycięzców i przegranych
    w trzech obszarach:

    - Wieku zawodników
    - Aktywności w stójce
    - Aktywności związanej z obaleniami

    Obalenia są ważne, ponieważ mogą zmieniać przebieg walki: przenoszą ją do parteru, umożliwiają kontrolę pozycji
    i mogą prowadzić do kolejnych ataków lub prób poddań.

    Ostatecznie raport pokazuje, że największa różnica między zwycięzcami i przegranymi była widoczna
    w liczbie znaczących uderzeń. Wiek również miał znaczenie, gdyż w analizowanym zbiorze młodsi zawodnicy
    wygrywali częściej niż starsi. Obalenia również były ważnym elementem analizy, ale ich związek z wynikiem 
    był mniej jednoznaczny.
    """)
