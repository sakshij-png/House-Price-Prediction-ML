from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# SETTINGS  (update TYPICAL_ERROR with your cross-validated MAE)
# ------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "house_price_model.pkl"
TYPICAL_ERROR = 970_000      # MAE of your best model, in rupees
DATASET_AVG = 4_766_729      # mean price in Housing.csv

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="centered")

# ------------------------------------------------------------------
# STYLE: beige and forest green (base colors live in .streamlit/config.toml)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&display=swap');
.block-container { padding-top: 2.5rem; max-width: 780px; }
h1, h2, h3, .hp-title, .hp-price { font-family: 'Fraunces', serif !important; color: #2A2018; }
.hp-head { display: flex; gap: 16px; align-items: center; margin-bottom: 1.4rem; }
.hp-mark { flex: none; width: 56px; height: 56px; border-radius: 14px; background: #2F5D50;
           display: flex; align-items: center; justify-content: center; }
.hp-title { font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: 0; }
.hp-sub { color: #6A5B4B; margin: .3rem 0 0; }
[data-testid="stForm"] { background: #FBF7EE; border: 1px solid #DDD1BA; border-radius: 14px; padding: 1.6rem; }
.hp-res { background: #22463C; color: #F3ECDF; border-radius: 14px; padding: 1.8rem; margin: 1.2rem 0; }
.hp-lbl { margin: 0; color: #CFC6B2; }
.hp-price { font-size: 3.2rem; font-weight: 700; color: #EBC878 !important; margin: .2rem 0 .3rem; line-height: 1.1; }
.hp-range { margin: 0; }
.hp-m3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
         border-top: 1px solid #4B6B60; margin-top: 1.3rem; padding-top: 1.1rem; }
.hp-m3 b { display: block; font-size: 1.2rem; }
.hp-m3 span { font-size: .85rem; color: #CFC6B2; }
@media (max-width: 600px) { .hp-m3 { grid-template-columns: 1fr; } }
#MainMenu, footer { visibility: hidden; }
</style>

<div class="hp-head">
  <div class="hp-mark">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#F3ECDF" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>
  </div>
  <div>
    <p class="hp-title">House Price Estimator</p>
    <p class="hp-sub">Enter a property's details and get an instant price estimate from a model trained on 545 listings.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"The model file could not be loaded from `{MODEL_PATH.name}`. Details: {e}")
    st.stop()


def fmt_inr(value: float) -> str:
    value = max(value, 0)
    if value >= 1e7:
        return f"₹ {value / 1e7:.2f} Cr"
    return f"₹ {value / 1e5:.2f} Lakh"


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def prepare_input(raw: dict) -> pd.DataFrame:
    """Works with the new Pipeline (raw columns) and the old encoded model."""
    cols = list(getattr(model, "feature_names_in_", []))
    if "mainroad" in cols:  # new pipeline trained on raw columns
        return pd.DataFrame([raw])[cols]

    encoded = {
        "area": raw["area"],
        "bedrooms": raw["bedrooms"],
        "bathrooms": raw["bathrooms"],
        "stories": raw["stories"],
        "parking": raw["parking"],
        "Bedrooms_per_area": raw["bedrooms"] / raw["area"],
        "furnishingstatus_semi-furnished": int(raw["furnishingstatus"] == "semi-furnished"),
        "furnishingstatus_unfurnished": int(raw["furnishingstatus"] == "unfurnished"),
    }
    for c in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        encoded[f"{c}_yes"] = int(raw[c] == "yes")
    return pd.DataFrame([encoded])[cols]


# ------------------------------------------------------------------
# INPUT FORM  (limits match the range of the training data)
# ------------------------------------------------------------------
with st.form("house_form"):
    st.subheader("Property details")
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq. ft.)", min_value=1650, max_value=16200, value=4600, step=50)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=1, step=1)
        parking = st.number_input("Parking spaces", min_value=0, max_value=3, value=0, step=1)
    with c2:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
        stories = st.number_input("Stories", min_value=1, max_value=4, value=2, step=1)
        furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"], index=1)

    st.subheader("Features")
    f1, f2, f3 = st.columns(3)
    mainroad = f1.checkbox("On a main road", value=True)
    guestroom = f2.checkbox("Guest room")
    basement = f3.checkbox("Basement")
    f4, f5, f6 = st.columns(3)
    hotwater = f4.checkbox("Hot water heating")
    aircon = f5.checkbox("Air conditioning")
    prefarea = f6.checkbox("Preferred area")

    submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# RESULT
# ------------------------------------------------------------------
if submitted:
    raw = {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "stories": stories, "parking": parking,
        "mainroad": yn(mainroad), "guestroom": yn(guestroom),
        "basement": yn(basement), "hotwaterheating": yn(hotwater),
        "airconditioning": yn(aircon), "prefarea": yn(prefarea),
        "furnishingstatus": furnishing.lower(),
    }
    try:
        prediction = float(model.predict(prepare_input(raw))[0])
    except Exception as e:
        st.error(f"The estimate could not be calculated. Details: {e}")
        st.stop()

    diff = (prediction / DATASET_AVG - 1) * 100
    st.markdown(
        f"""
<div class="hp-res">
  <p class="hp-lbl">Estimated price</p>
  <p class="hp-price">{fmt_inr(prediction)}</p>
  <p class="hp-range">Likely range: {fmt_inr(prediction - TYPICAL_ERROR)} to {fmt_inr(prediction + TYPICAL_ERROR)}</p>
  <div class="hp-m3">
    <div><b>₹ {prediction:,.0f}</b><span>Exact estimate</span></div>
    <div><b>₹ {prediction / area:,.0f}</b><span>Price per sq. ft.</span></div>
    <div><b>{diff:+.0f}%</b><span>Vs. average house ({fmt_inr(DATASET_AVG)})</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------------
with st.expander("About this estimate"):
    st.write(
        f"The model is a scikit-learn regression model trained on 545 house listings "
        f"(area, rooms, stories, parking, amenities and furnishing). On unseen data its "
        f"typical error is about {fmt_inr(TYPICAL_ERROR)}, so treat the result as a guide, "
        f"not a valuation. Inputs are limited to the ranges found in the training data, "
        f"because the model can't predict reliably outside them."
    )

st.caption("Built with Python, pandas, scikit-learn and Streamlit.")