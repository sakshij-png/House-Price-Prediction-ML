# 🏠 House Price Estimator

A machine learning web app that estimates a house's price from its size, rooms, amenities and furnishing, built with Python, scikit-learn and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

**🔗 Live app:** <!-- UPDATE: paste your Streamlit link --> [[Open the House Price Estimator](https://YOUR-APP-LINK.streamlit.app](https://house-price-prediction-ml-lwpc6gsgnpcqsxo3bpvw7a.streamlit.app/)

<img width="1275" height="853" alt="Screenshot 2026-11-01 121902" src="https://github.com/user-attachments/assets/8a36e1a4-fe1c-4f13-9170-8f88def77db1" />


App screenshot

## What it does

Enter a property's details and the app returns:

- the **estimated price**, shown in Lakh/Crore
- a **likely price range**, based on the model's typical error
- the **price per sq. ft.** and how the estimate compares with the average house in the data

Inputs are limited to the ranges found in the training data, so the model is never asked to predict far outside what it has seen.

## Dataset

The project uses the public **Housing Prices** dataset (`Housing.csv`) from Kaggle: **545 listings** with 12 features and the price.

| Type | Features |
|---|---|
| Numeric | `area`, `bedrooms`, `bathrooms`, `stories`, `parking` |
| Yes/No | `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` |
| Category | `furnishingstatus` (furnished, semi-furnished, unfurnished) |

Prices are shown in rupees (₹).

## Approach

1. **Exploration:** price distribution, area vs. price, bedrooms vs. price and a correlation heatmap.
2. **Cleaning:** checked for missing values and duplicates (none found).
3. **Feature engineering:** added `Bedrooms_per_area` and one-hot encoded the categorical columns.
4. **Avoiding data leakage:** a `Price_per_area` feature was tried and removed, because it is calculated from the target (price) and would make the scores look better than they really are.
5. **Modeling:** an 80/20 train/test split, comparing Linear Regression, Ridge Regression and Random Forest.
6. **Deployment:** the trained model is saved with `joblib` and served through a Streamlit app.

## Results

Scores on the 109-house test set:

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | 986,563 | 1,345,327 | 0.642 |
| **Ridge Regression** | **970,246** | **1,325,320** | **0.652** |
| Random Forest | 1,019,553 | 1,394,397 | 0.615 |

<!-- UPDATE: after you run cross-validation, replace this table with the 5-fold mean ± std results -->

**What drives the price?** In the Random Forest, **area** is by far the most important feature (about 44% of importance), followed by **bathrooms** (about 15%). Air conditioning, stories and parking come next.

A typical prediction is off by roughly **₹10 Lakh**, which is why the app shows a price range and not just a single number.

## Run it locally

```bash
# 1. Clone the repository
git clone https://github.com/sakshij-png/House-Price-Prediction-ML.git
cd House-Price-Prediction-ML

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
House-Price-Prediction-ML/
├── app.py                  # Streamlit web app
├── data/
│   └── Housing.csv         # Dataset
├── models/
│   └── house_price_model.pkl   # Trained model
├── notebooks/
│   └── house_price_analysis.ipynb   # EDA, modeling and evaluation
├── .streamlit/
│   └── config.toml         # App theme
├── requirements.txt
└── README.md
```

## Key learnings

- **Data leakage is easy to miss.** A feature built from the target (`Price_per_area`) made the scores look great. Removing it gave honest results.
- **A simpler model can win on small data.** With 545 rows, Ridge Regression scored better than Random Forest, so more complex does not always mean better.
- **One metric isn't enough.** MAE shows the typical error in rupees, RMSE punishes big misses, and R² shows how much of the price variation the model explains.
- **Feature importance explains the model.** Area drives the price far more than any other feature, which matches common sense.
- **Deployment needs guardrails.** Limiting inputs to the training range and showing a price range makes the app more trustworthy than a single exact number.
- **Shipping is its own skill.** Saving the model, building a clean Streamlit interface and publishing it taught me more than the notebook alone.

## Author

**Sakshi Joshi**

B.Tech student in AI & Data Science
