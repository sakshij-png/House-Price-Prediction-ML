import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏠 House Price Prediction")
st.write(
    "Enter the details of the house below to estimate its price "
    "using a Machine Learning model."
)

st.divider()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "models/house_price_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Could not load the machine learning model.")
    st.error(f"Error: {e}")
    st.stop()


# --------------------------------------------------
# HOUSE DETAILS
# --------------------------------------------------

st.subheader("🏡 Enter House Details")


col1, col2, col3 = st.columns(3)


with col1:

    area = st.number_input(
        "Area (sq. ft.)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


with col2:

    stories = st.number_input(
        "Number of Stories",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )

    mainroad = st.selectbox(
        "Main Road Access",
        ["Yes", "No"]
    )


with col3:

    guestroom = st.selectbox(
        "Guest Room",
        ["Yes", "No"]
    )

    basement = st.selectbox(
        "Basement",
        ["Yes", "No"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["Yes", "No"]
    )


st.divider()


# --------------------------------------------------
# ADDITIONAL FEATURES
# --------------------------------------------------

st.subheader("✨ Additional Features")


col4, col5, col6 = st.columns(3)


with col4:

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["Yes", "No"]
    )


with col5:

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["Yes", "No"]
    )


with col6:

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        [
            "Furnished",
            "Semi-Furnished",
            "Unfurnished"
        ]
    )


st.divider()


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

# IMPORTANT:
# Price_per_area is NOT used because it directly
# depends on the target variable (price).
#
# Bedrooms_per_area does NOT use price, so it is safe.

bedrooms_per_area = bedrooms / area


# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame({

    "area": [area],

    "bedrooms": [bedrooms],

    "bathrooms": [bathrooms],

    "stories": [stories],

    "parking": [parking],

    "Bedrooms_per_area": [bedrooms_per_area],

    "mainroad_yes": [
        1 if mainroad == "Yes" else 0
    ],

    "guestroom_yes": [
        1 if guestroom == "Yes" else 0
    ],

    "basement_yes": [
        1 if basement == "Yes" else 0
    ],

    "hotwaterheating_yes": [
        1 if hotwaterheating == "Yes" else 0
    ],

    "airconditioning_yes": [
        1 if airconditioning == "Yes" else 0
    ],

    "prefarea_yes": [
        1 if prefarea == "Yes" else 0
    ],

    "furnishingstatus_semi-furnished": [
        1 if furnishingstatus == "Semi-Furnished" else 0
    ],

    "furnishingstatus_unfurnished": [
        1 if furnishingstatus == "Unfurnished" else 0
    ]
})


# --------------------------------------------------
# MAKE SURE INPUT COLUMNS MATCH THE MODEL
# --------------------------------------------------

expected_columns = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "Bedrooms_per_area",
    "mainroad_yes",
    "guestroom_yes",
    "basement_yes",
    "hotwaterheating_yes",
    "airconditioning_yes",
    "prefarea_yes",
    "furnishingstatus_semi-furnished",
    "furnishingstatus_unfurnished"
]


# Reorder columns exactly as expected
input_data = input_data[expected_columns]


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.subheader("🔮 Predict House Price")


if st.button(
    "Predict House Price",
    type="primary",
    use_container_width=True
):

    try:

        prediction = model.predict(input_data)[0]

        st.success(
            f"### 🏠 Estimated House Price: ₹ {prediction:,.2f}"
        )

        st.info(
            "The prediction is generated using the trained "
            "Random Forest Regression model."
        )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.error(f"Error: {e}")

        st.write("Input columns:")
        st.write(input_data.columns.tolist())


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📊 About the Project")

st.write(
    """
    This project uses Machine Learning to predict house prices
    based on various property characteristics.

    **Machine Learning Model:** Random Forest Regression

    **Features used:**
    - Area
    - Bedrooms
    - Bathrooms
    - Stories
    - Parking
    - Main road access
    - Guest room
    - Basement
    - Hot water heating
    - Air conditioning
    - Preferred area
    - Furnishing status
    - Bedrooms per area
    """
)

st.caption(
    "House Price Prediction ML Project | Python • Pandas • Scikit-learn • Streamlit"
)