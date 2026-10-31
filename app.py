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
# STYLE: blueprint-style header and result panel
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&display=swap');
.block-container { padding-top: 2rem; max-width: 820px; }
h1, h2, h3, .bp-title, .bp-price { font-family: 'Bricolage Grotesque', sans-serif !important; }
.bp-hero {
    background-color: #14324A;
    background-image:
        linear-gradient(rgba(255,255,255,.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px);
    background-size: 28px 28px;
    border-radius: 14px; padding: 2rem 2rem 1.6rem; margin-bottom: 1.2rem; color: #F4F7FA;
}
.bp-title { font-size: 2.2rem; font-weight: 700; margin: 0 0 .3rem; line-height: 1.15; }
.bp-sub { font-size: 1.02rem; opacity: .85; margin: 0; max-width: 34rem; }
.bp-result {
    background-color: #14324A;
    background-image:
        linear-gradient(rgba(255,255,255,.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px);
    background-size: 28px 28px;
    border: 2px dashed #F2B134; border-radius: 14px;
    padding: 1.6rem 1.8rem; margin: 1rem 0; color: #F4F7FA;
}
.bp-label { font-size: .95rem; opacity: .8; margin: 0; }
.bp-price { font-size: 3rem; font-weight: 700; color: #F2B134; margin: .1rem 0 .2rem; line-height: 1.1; }
.bp-range { font-size: 1rem; margin: 0; opacity: .9; }
#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="bp-hero">
  <p class="bp-title">What is this house worth?</p>
  <p class="bp-sub">Describe the property and get an instant price estimate
  from a machine learning model trained on 545 listings.</p>
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
        area = st.slider("Area (sq. ft.)", 1650, 16200, 4600, step=50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 1)
    with c2:
        stories = st.slider("Stories", 1, 4, 2)
        parking = st.slider("Parking spaces", 0, 3, 0)
        furnishing = st.selectbox(
            "Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"], index=1
        )

    st.subheader("Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        mainroad = st.checkbox("On a main road", value=True)
        guestroom = st.checkbox("Guest room")
    with f2:
        basement = st.checkbox("Basement")
        hotwater = st.checkbox("Hot water heating")
    with f3:
        aircon = st.checkbox("Air conditioning")
        prefarea = st.checkbox("Preferred area")

    submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


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

    st.markdown(
        f"""
<div class="bp-result">
  <p class="bp-label">Estimated price</p>
  <p class="bp-price">{fmt_inr(prediction)}</p>
  <p class="bp-range">Likely range: {fmt_inr(prediction - TYPICAL_ERROR)} to {fmt_inr(prediction + TYPICAL_ERROR)}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Exact estimate", f"₹ {prediction:,.0f}")
    m2.metric("Price per sq. ft.", f"₹ {prediction / area:,.0f}")
    m3.metric(
        "Vs. average house in data",
        fmt_inr(DATASET_AVG),
        delta=f"{(prediction / DATASET_AVG - 1) * 100:+.0f}%",
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