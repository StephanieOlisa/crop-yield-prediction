
import streamlit as st
import pandas as pd
import pickle

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM DARK GREEN DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #10291F;
    color: #F1F7F1;
}

[data-testid="stHeader"] {
    background-color: #10291F;
}

[data-testid="stSidebar"] {
    background-color: #183D2D;
}

[data-testid="stSidebar"] * {
    color: #F1F7F1 !important;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #A5D6A7;
    margin-bottom: 5px;
}

.subtitle {
    color: #B7CDBB;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    color: #A5D6A7;
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
}

.info-card, .pest-card {
    background: #1C3B2D;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #355B43;
    margin-bottom: 15px;
    color: #F1F7F1;
}

.result-card {
    background: #214A34;
    padding: 25px;
    border-radius: 16px;
    border-left: 6px solid #81C784;
    margin: 15px 0;
}

.small-note {
    color: #B7CDBB;
    font-size: 13px;
}

[data-testid="stMetric"] {
    background: #1C3B2D;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #355B43;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #F1F7F1 !important;
}

.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background-color: #388E3C;
    color: white;
}

.stTextInput input,
.stNumberInput input {
    background-color: #1C3B2D;
    color: white;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #1C3B2D;
}

div[data-baseweb="select"] * {
    color: white;
}

.stMarkdown, p, li, label {
    color: #E4EEE5;
}

hr {
    border-color: #355B43;
}

[data-testid="stDataFrame"] {
    background-color: #1C3B2D;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATA AND MACHINE LEARNING MODEL
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_csv("yield_df.csv")

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    return data


@st.cache_resource
def load_models():

    with open("crop_yield_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


try:

    df = load_data()
    model, preprocessor = load_models()

except FileNotFoundError as e:

    st.error(f"Required project file not found: {e.filename}")

    st.info("""
    Ensure these files are in the same folder as app.py:

    - yield_df.csv
    - crop_yield_model.pkl
    - preprocessor.pkl
    """)

    st.stop()

except Exception as e:

    st.error(f"Unable to load application files: {e}")
    st.stop()


if "Item" not in df.columns:

    st.error("Your dataset must contain an Item column.")
    st.stop()


item_list = sorted(df["Item"].dropna().unique().tolist())


# =====================================================
# 4. CROP PEST AND MANAGEMENT DATABASE
# =====================================================

crop_information = {

    "Maize": {
        "season": "April - July",
        "pests": [
            "Fall Armyworm",
            "Stem Borers",
            "Maize Weevils",
            "Aphids"
        ],
        "damage": "Fall armyworms feed on maize leaves and growing points. Stem borers tunnel into stems, while weevils damage stored grains.",
        "symptoms": "Holes in leaves, damaged plant funnels, sawdust-like waste, broken stems and damaged grains.",
        "control": "Inspect plants regularly, remove heavily infested plant parts, practice crop rotation and maintain field sanitation.",
        "treatment": "Consider biological control such as Bacillus thuringiensis for suitable young caterpillars. Use a registered insecticide only when its label permits use on maize against the identified pest."
    },

    "Rice": {
        "season": "May - August",
        "pests": [
            "Rice Stem Borers",
            "Rice Gall Midge",
            "Rice Leafhoppers",
            "Rice Hispa"
        ],
        "damage": "Stem borers damage rice stems, gall midges affect new shoots, and leafhoppers suck sap from plants.",
        "symptoms": "Dead central shoots, white empty panicles, twisted shoots and yellowing leaves.",
        "control": "Use resistant varieties, remove affected plants, maintain appropriate water management and avoid excessive nitrogen fertilizer.",
        "treatment": "Use only a registered rice pesticide labelled for the identified pest. Confirm its application rate and pre-harvest interval."
    },

    "Cassava": {
        "season": "March - June",
        "pests": [
            "Cassava Mealybugs",
            "Whiteflies",
            "Cassava Green Mites",
            "Termites"
        ],
        "damage": "Mealybugs and mites damage leaves and growing points. Whiteflies suck plant sap and may transmit cassava mosaic disease.",
        "symptoms": "Curling leaves, yellowing, stunted growth, distorted shoots and damaged stems.",
        "control": "Plant healthy cassava cuttings, remove severely affected plants, control weeds and use resistant varieties.",
        "treatment": "Prioritize clean planting materials, resistant varieties and biological control. Use only a product registered for cassava and the identified pest."
    },

    "Yams": {
        "season": "March - May",
        "pests": [
            "Yam Beetles",
            "Yam Nematodes",
            "Yam Scale Insects",
            "Termites"
        ],
        "damage": "Yam beetles feed on developing tubers, nematodes cause tuber damage, and termites attack tubers and plant tissues.",
        "symptoms": "Holes in tubers, rotting, cracked tuber surfaces, poor sprouting and stunted growth.",
        "control": "Use healthy seed yams, select clean planting sites, practice crop rotation and remove infected plant residues.",
        "treatment": "Use approved seed or soil treatment products only when specifically labelled for yam and the identified pest."
    },

    "Plantains": {
        "season": "Year-round with adequate rainfall",
        "pests": [
            "Banana Weevils",
            "Nematodes",
            "Banana Aphids",
            "Mealybugs"
        ],
        "damage": "Banana weevils tunnel into plant corms, nematodes damage roots, and aphids suck sap and may transmit viral diseases.",
        "symptoms": "Weak plants, yellow leaves, poor bunch development, damaged roots and plants falling over.",
        "control": "Use clean planting materials, remove old infected plant residues, maintain field sanitation and practice crop rotation.",
        "treatment": "Use clean planting materials and biological or cultural controls first. Chemical products must be registered for plantain and the target pest."
    },

    "Potatoes": {
        "season": "October - February",
        "pests": [
            "Potato Tuber Moth",
            "Aphids",
            "Cutworms",
            "Wireworms"
        ],
        "damage": "Tuber moth larvae damage leaves and tubers, aphids suck plant sap, and wireworms feed on underground tubers.",
        "symptoms": "Holes in leaves, tunnels inside tubers, wilting and damaged underground stems.",
        "control": "Use healthy seed potatoes, earth up plants, remove infected tubers and avoid leaving harvested tubers exposed.",
        "treatment": "Use a registered potato insecticide appropriate for the identified pest. Observe the product label and pre-harvest interval."
    },

    "Sweet Potatoes": {
        "season": "April - July",
        "pests": [
            "Sweet Potato Weevils",
            "Whiteflies",
            "Aphids",
            "Wireworms"
        ],
        "damage": "Sweet potato weevils tunnel into roots and vines, while whiteflies and aphids suck sap from leaves.",
        "symptoms": "Tunnels in roots, bitter or damaged tubers, yellowing leaves and weak vines.",
        "control": "Use healthy vines, rotate crops, remove infested roots and vines, and harvest promptly.",
        "treatment": "Use a registered product specifically labelled for sweet potato and the identified pest. Prioritize cultural control and clean planting materials."
    },

    "Soybeans": {
        "season": "June - August",
        "pests": [
            "Aphids",
            "Armyworms",
            "Pod Borers",
            "Stink Bugs"
        ],
        "damage": "Aphids suck sap, armyworms eat leaves, and pod borers and stink bugs damage developing pods and seeds.",
        "symptoms": "Curled leaves, missing leaf tissue, damaged pods and poorly developed seeds.",
        "control": "Scout plants regularly, encourage beneficial insects, remove heavily infested residues and practice crop rotation.",
        "treatment": "Use a registered soybean insecticide only when pest populations justify treatment. Select a product labelled for the target pest."
    },

    "Sorghum": {
        "season": "May - July",
        "pests": [
            "Sorghum Stem Borers",
            "Sorghum Midge",
            "Armyworms",
            "Aphids"
        ],
        "damage": "Stem borers tunnel into stems, sorghum midge damages developing grains, and aphids suck plant sap.",
        "symptoms": "Dead shoots, damaged stems, poorly filled grains and yellowing leaves.",
        "control": "Plant at recommended times, remove crop residues, rotate crops and monitor flowering heads.",
        "treatment": "Use a locally registered product labelled for sorghum and the specific pest. Treatment should be based on field monitoring."
    },

    "Wheat": {
        "season": "November - January",
        "pests": [
            "Aphids",
            "Armyworms",
            "Hessian Flies",
            "Wireworms"
        ],
        "damage": "Aphids suck sap and may transmit viruses, armyworms eat leaves, and wireworms attack roots and seedlings.",
        "symptoms": "Yellow leaves, damaged seedlings, holes in leaves and poorly developed plants.",
        "control": "Use healthy seeds, practice crop rotation, remove volunteer wheat plants and monitor crops regularly.",
        "treatment": "Use a registered wheat pesticide specifically labelled for the identified pest. Follow the product application instructions."
    }
}


# =====================================================
# 5. YIELD INTERPRETATION RANGES
# =====================================================

yield_ranges = {

    "Potatoes": (100000, 350000),
    "Cassava": (70000, 280000),
    "Sweet Potatoes": (60000, 250000),
    "Yams": (50000, 180000),
    "Plantains": (50000, 280000),
    "Maize": (15000, 100000),
    "Rice": (15000, 70000),
    "Wheat": (12000, 60000),
    "Sorghum": (8000, 80000),
    "Soybeans": (7000, 30000)

}


# =====================================================
# 6. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("# 🌾 Crop Yield Prediction")

    st.caption("Smart Agricultural Decision Support")

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "Select a section",
        [
            "Yield Prediction",
            "Crop Pest Guide",
            "Planting Calendar",
            "Yield Reference Guide"
        ]
    )

    st.markdown("---")

    st.markdown("### About This Application")

    st.write("""
    Crop Yield Prediction uses machine learning and
    agricultural information to support better
    crop production decisions.

    It provides yield estimates and crop management
    recommendations based on selected growing conditions.
    """)

    st.markdown("---")

    st.caption("Data Science | Agriculture | Machine Learning")


# =====================================================
# 7. MAIN APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🌾 Crop Yield Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Agricultural Insights for Better Crop Production</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# 8. YIELD PREDICTION PAGE
# =====================================================

if page == "Yield Prediction":

    st.markdown(
        '<div class="section-title">🌱 Crop Yield Prediction</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Enter the expected growing conditions and select a crop
    to estimate its potential yield.
    """)

    st.markdown("### 🌦️ Environmental Conditions")

    col1, col2 = st.columns(2)

    with col1:

        rainfall = st.number_input(
            "Average Rainfall (mm/year)",
            min_value=0.0,
            value=1200.0,
            step=50.0
        )

        pesticides = st.number_input(
            "Estimated Pesticide Usage (tonnes)",
            min_value=0.0,
            value=50000.0,
            step=100.0,
            help="""
            Enter pesticide usage using the same units and
            scale as the model training dataset.

            This is a machine learning input, not the actual
            pesticide dosage for a farm.
            """
        )

    with col2:

        avg_temp = st.number_input(
            "Average Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.5
        )

        item = st.selectbox(
            "Select Crop Type",
            item_list
        )

    st.markdown("---")

    if st.button("🌾 Predict Crop Yield"):

        input_df = pd.DataFrame({

            "average_rain_fall_mm_per_year": [rainfall],

            "pesticides_tonnes": [pesticides],

            "avg_temp": [avg_temp],

            "Item": [item]

        })

        try:

            transformed_data = preprocessor.transform(input_df)

            prediction = model.predict(transformed_data)

            yield_value = float(prediction[0])

            st.session_state["yield_value"] = yield_value

            st.session_state["selected_crop"] = item

            st.markdown("---")

            st.markdown("## 📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Selected Crop",
                    item
                )

            with col2:

                st.metric(
                    "Predicted Yield",
                    f"{yield_value:,.2f} hg/ha"
                )

            with col3:

                st.metric(
                    "Average Temperature",
                    f"{avg_temp:.1f} °C"
                )

            # -----------------------------------------
            # YIELD INTERPRETATION
            # -----------------------------------------

            if item in yield_ranges:

                low, high = yield_ranges[item]

                if yield_value < low:

                    st.error("🔴 LOW YIELD")

                    st.write("""
                    The predicted yield is below the reference
                    range for this crop.

                    **Recommendations:**

                    - Review soil fertility.
                    - Improve water management.
                    - Check for pests and diseases.
                    - Review planting materials.
                    - Improve farm management practices.
                    """)

                elif yield_value < high:

                    st.warning("🟡 AVERAGE YIELD")

                    st.write("""
                    The predicted yield falls within the
                    reference range for this crop.

                    **Recommendations:**

                    - Continue good agricultural practices.
                    - Monitor pests and diseases.
                    - Maintain suitable soil moisture.
                    - Improve nutrient management where necessary.
                    """)

                else:

                    st.success("🟢 HIGH YIELD")

                    st.write("""
                    The predicted yield is above the reference
                    range for this crop.

                    **Recommendations:**

                    - Maintain effective crop management.
                    - Continue monitoring pests and diseases.
                    - Maintain soil fertility.
                    - Protect the crop until harvest.
                    """)

                # -------------------------------------
                # YIELD PERFORMANCE INDICATOR
                # -------------------------------------

                st.markdown("### Yield Performance Indicator")

                progress = max(
                    0.0,
                    min(yield_value / high, 1.0)
                )

                st.progress(progress)

                st.caption(
                    f"Reference high-yield threshold: {high:,.0f} hg/ha"
                )

            else:

                st.info(
                    "A reference yield range is not currently "
                    "available for this crop."
                )

            # -------------------------------------
            # GROWING CONDITIONS SUMMARY
            # -------------------------------------

            st.markdown("---")

            st.markdown("### 🌦️ Growing Conditions Summary")

            summary1, summary2, summary3 = st.columns(3)

            summary1.metric(
                "Rainfall",
                f"{rainfall:,.1f} mm"
            )

            summary2.metric(
                "Temperature",
                f"{avg_temp:.1f} °C"
            )

            summary3.metric(
                "Pesticide Input",
                f"{pesticides:,.1f} tonnes"
            )

            # -------------------------------------
            # CROP MANAGEMENT GUIDE
            # -------------------------------------

            if item in crop_information:

                crop = crop_information[item]

                st.markdown("---")

                st.markdown("## 🌱 Crop Management Recommendations")

                st.info(
                    f"**Recommended planting period:** {crop['season']}"
                )

                pest_col, management_col = st.columns(2)

                with pest_col:

                    st.markdown("### 🐛 Common Pests")

                    for pest in crop["pests"]:

                        st.markdown(f"- {pest}")

                with management_col:

                    st.markdown("### 🔍 Pest Symptoms")

                    st.write(crop["symptoms"])

                st.markdown("### ⚠️ How Pests Affect This Crop")

                st.write(crop["damage"])

                st.markdown("### 🌿 Prevention and Control")

                st.write(crop["control"])

                st.markdown("### 🧪 Treatment Recommendations")

                st.write(crop["treatment"])

                st.warning("""
                **Pesticide Safety**

                This application does not prescribe a universal
                pesticide quantity.

                The actual application rate must come from the
                label of a registered product approved for the
                crop and identified pest.

                Check the active ingredient, concentration,
                formulation, application method, farm area,
                protective equipment and pre-harvest interval.

                Do not increase pesticide application to improve
                predicted crop yield.
                """)

        except Exception as e:

            st.error(
                f"An error occurred during prediction: {e}"
            )

    # ---------------------------------------------
    # LAST PREDICTION
    # ---------------------------------------------

    if "yield_value" in st.session_state:

        st.markdown("---")

        st.caption(
            f"Last prediction: {st.session_state['selected_crop']} | "
            f"{st.session_state['yield_value']:,.2f} hg/ha"
        )


# =====================================================
# 9. CROP PEST GUIDE PAGE
# =====================================================

elif page == "Crop Pest Guide":

    st.markdown(
        '<div class="section-title">🐛 Crop Pest Identification and Management</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Explore common pests affecting different crops,
    the damage they cause, their symptoms and
    recommended management practices.
    """)

    selected_crop = st.selectbox(
        "Choose a crop to explore",
        item_list,
        key="pest_crop"
    )

    if selected_crop in crop_information:

        crop = crop_information[selected_crop]

        st.markdown("---")

        st.markdown(f"## 🌾 {selected_crop}")

        st.info(
            f"Recommended planting period: {crop['season']}"
        )

        st.markdown("### 🐛 Common Pests")

        for index, pest in enumerate(crop["pests"], 1):

            st.markdown(
                f"""
                <div class="pest-card">
                    <strong>🐛 {index}. {pest}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### ⚠️ Crop Damage")

        st.write(crop["damage"])

        st.markdown("### 🔍 Symptoms")

        st.write(crop["symptoms"])

        st.markdown("### 🌿 Prevention and Control")

        st.write(crop["control"])

        st.markdown("### 🧪 Treatment Recommendations")

        st.write(crop["treatment"])

        st.warning("""
        Identify the pest correctly before treatment.
        Consult an agricultural extension officer and
        follow registered pesticide product labels.
        """)

    else:

        st.info(
            "Pest information for this crop has not yet been added."
        )


# =====================================================
# 10. PLANTING CALENDAR PAGE
# =====================================================

elif page == "Planting Calendar":

    st.markdown(
        '<div class="section-title">📅 Crop Planting Calendar</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Explore general planting periods for the crops
    available in this application.
    """)

    calendar_data = []

    for crop_name, details in crop_information.items():

        calendar_data.append({

            "Crop": crop_name,

            "Recommended Planting Period": details["season"]

        })

    calendar_df = pd.DataFrame(calendar_data)

    st.dataframe(
        calendar_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown("### 🌱 Planting Advice")

    st.write("""
    - Check local rainfall forecasts before planting.
    - Use certified or healthy planting materials.
    - Prepare the land before the planting season.
    - Consider soil type and drainage.
    - Consult local agricultural extension services
      for location-specific recommendations.
    """)

    st.info("""
    Planting periods are general guides. Actual planting
    dates depend on location, rainfall, irrigation and
    crop variety.
    """)


# =====================================================
# 11. YIELD REFERENCE GUIDE PAGE
# =====================================================

elif page == "Yield Reference Guide":

    st.markdown(
        '<div class="section-title">📊 Crop Yield Reference Guide</div>',
        unsafe_allow_html=True
    )

    st.write("""
    This table provides the reference yield thresholds
    used by the application to interpret predictions.
    """)

    guide_data = []

    for crop_name, values in yield_ranges.items():

        low, high = values

        guide_data.append({

            "Crop": crop_name,

            "Low Yield (hg/ha)": f"Below {low:,.0f}",

            "Average Reference Range (hg/ha)": (
                f"{low:,.0f} - {high:,.0f}"
            ),

            "High Yield (hg/ha)": f"Above {high:,.0f}"

        })

    guide_df = pd.DataFrame(guide_data)

    st.dataframe(
        guide_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.caption("""
    These are application reference thresholds, not universal
    agricultural standards. They should be validated against
    crop-specific, location-specific and farm-level data.
    """)


# =====================================================
# 12. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:20px;">

    <h3 style="color:#A5D6A7;">
    🌾 Crop Yield Prediction
    </h3>

    <p style="color:#B7CDBB;">
    Using Data Science and Machine Learning to Support
    Better Agricultural Decisions.
    </p>

    <p style="color:#8A978D; font-size:12px;">
    Crop Yield Prediction | Agricultural Decision Support
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("Important Disclaimer"):

    st.write("""
    Crop Yield Prediction provides estimated crop yields using
    a trained machine learning model and environmental inputs.

    Predictions are not guaranteed actual harvest outcomes.

    Results may vary due to soil fertility, farm management,
    irrigation, crop variety, weather, pests, diseases and
    other environmental conditions.

    Pest information is provided as general guidance.
    This application does not diagnose an infestation or
    prescribe pesticide dosages.

    Always consult qualified agricultural extension officers
    and follow local pesticide registration and safety
    requirements before applying agricultural chemicals.
    """)

