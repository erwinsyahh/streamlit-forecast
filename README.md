# 📈 Streamlit Forecast Dashboard

A modular Streamlit app for time series forecasting with interactive visualizations, exploratory data analysis, and model inference using Snowflake data.

## 🚀 Features

* **📄 Multi-page navigation** with:

  * Dashboard overview
  * Exploratory analysis
  * Partition preview and decomposition
  * Forecast visualizations
  * Stationarity checks and statistical testing
* **⚙️ Modular structure** using `utils/` for clean preprocessing and caching
* **📊 Rich interactive charts** with Plotly and Matplotlib
* **📦 Snowflake integration** via Snowpark
* **🔬 ADF test, decomposition, correlation matrix, ACF/PACF** by partition
* **📉 Forecast results** with volume predictions

## 🗂️ Folder Structure

```
streamlit-forecast/
├── .venv/
├── notebook/
│   ├── exploration.ipynb
├── pages/
│   ├── 📊 Page_1_-_Data_Explorer.py
│   ├── 📦 Page_2_-_Training_Result.py
│   ├── 🔎 Page_3_-_Model_Performance_Summary.py
│   ├── 📈 Page_4_-_Feature_importance.py
├── utils/
│   ├── load_data.py
├── Home.py
├── requirements.txt
└── README.md
```

## 💪 Setup

1. **Clone the repo**

```bash
git clone https://github.com/erwinsyahh/streamlit-forecast.git
cd streamlit-forecast
```

2. **Install dependencies (via `uv` or `pip`)**

```bash
uv pip install -r requirements.txt
```

*or*

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run Home.py
```

## ⚙️ Configuration

* Configure your **Snowflake connection** locally or directly in Snowflake
* Filtering and date ranges can be adjusted through the sidebar UI
* Uses default **30-day** period for decomposition

## 📌 Dependencies

* `streamlit`
* `pandas`, `numpy`
* `scikit-learn`
* `plotly`, `matplotlib`, `seaborn`
* `statsmodels`
* `snowflake-snowpark-python`

## 💌 Contributions

Feel free to fork and customize for other time series projects or different backends.

---

Built with ❤️ by [@erwinsyahh](https://github.com/erwinsyahh)
