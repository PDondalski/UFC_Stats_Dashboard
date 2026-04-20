# 🥊 UFC Fight Results Dashboard

An interactive exploratory dashboard built with **Streamlit**.  
The project analyzes UFC fight results using data from Kaggle, originally collected from the official UFCStats website (data as of November 2024).

The main goal of the application is to identify which statistical factors are most commonly associated with winning a fight. The analysis focuses on fighter age, significant strikes, and takedown attempts and effectiveness.

---

## 🔍 Scope of Analysis

The report focuses on comparing winners and losers.  
The dataset was transformed from a technical `f1` / `f2` structure into a more interpretable `winner` / `loser` perspective, since the original fighter ordering does not carry meaningful analytical information.

The analysis includes:

- number of fights across main weight classes,
- relationship between age differences and fight outcomes,
- comparison of significant strikes between winners and losers,
- takedown attempts and successful takedowns,
- a simple regression model predicting the number of successful takedowns.

---

## ⚙️ Features

- navigation between report sections,
- filtering by weight class,
- filtering by age difference,
- interactive Plotly visualizations,
- dynamic data table,
- linear regression model with R² and MAE metrics,
- input form for testing predictions.

---

## 🛠️ Technologies

- Python 3.10
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## 🧠 Application Structure

The dashboard is divided into four main sections:

1. **Introduction** – overview of the dataset, project goal, and analysis scope.  
2. **Data Exploration** – interactive analysis of fights, age differences, striking, and takedowns.  
3. **Model** – linear regression model predicting successful takedowns.  
4. **Conclusions** – summary of the most important findings.

---

## 📎 Data Source

Dataset:  
https://www.kaggle.com/datasets/thasankakandage/ufc-dataset-2024

The dataset contains UFC fight statistics and reflects data available as of November 2024.  
The data was originally sourced from the official UFCStats website.
