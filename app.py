import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# Load files
model = joblib.load("podium_model.pkl_final")
driver_encoder = joblib.load("driver_encoder.pkl_final")
team_encoder = joblib.load("team_encoder.pkl_final")
driver_stats = pd.read_csv("driver_stats.csv")

st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 100%;
}

.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #30363D;
}

.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 1.1rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='big-title'>
🏎️ Formula 1 Podium Predictor
</div>

<div class='subtitle'>
Machine Learning powered Formula 1 analytics dashboard
</div>
""", unsafe_allow_html=True)

st.write(
    "Predict whether a Formula 1 driver is likely to finish on the podium."
)

left, right = st.columns([1, 1])

with left:
    st.subheader("🏁 Race Configuration")

    driver = st.selectbox(
        "Driver",
        sorted(driver_encoder.classes_)
    )

    team = st.selectbox(
        "Team",
        sorted(team_encoder.classes_)
    )

    grid_position = st.slider(
        "Grid Position",
        1,
        20,
        5
    )

    predict_clicked = st.button(
        "🚀 Predict Podium Probability",
        use_container_width=True
    )

with right:
    stats = driver_stats[
        driver_stats["Driver"] == driver
    ]

    avg_points = stats["AvgDriverPoints"].iloc[0]
    avg_finish = stats["AvgDriverFinish"].iloc[0]

    st.markdown(f"""
    <div style="
    background:#161B22;
    padding:25px;
    border-radius:15px;
    border:1px solid #30363D;
    ">
        <h2>🏎️ {driver}</h2>
        <h4>⭐ Average Points: {avg_points:.2f}</h4>
        <h4>🏁 Average Finish Position: {avg_finish:.2f}</h4>
    </div>
    """, unsafe_allow_html=True)

if predict_clicked:

    stats = driver_stats[
        driver_stats["Driver"] == driver
    ]

    avg_points = stats["AvgDriverPoints"].iloc[0]
    avg_finish = stats["AvgDriverFinish"].iloc[0]

    input_data = pd.DataFrame({
        "GridPosition": [grid_position],
        "DriverEncoded": [
            driver_encoder.transform([driver])[0]
        ],
        "TeamEncoded": [
            team_encoder.transform([team])[0]
        ],
        "AvgDriverPoints": [avg_points],
        "AvgDriverFinish": [avg_finish]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
        "Podium Probability",
        f"{probability*100:.2f}%"
    )

    with col2:
        st.metric(
        "Avg Points",
        f"{avg_points:.2f}"
    )

    with col3:
        st.metric(
        "Avg Finish",
        f"{avg_finish:.2f}"
    )


    if probability >= 0.80:
        st.success("🏆 Very Strong Podium Candidate")

    elif probability >= 0.60:
        st.info("🥈 Good Chance of Podium")

    elif probability >= 0.40:
        st.warning("⚠️ Outside Chance")

    else:
        st.error("❌ Podium Unlikely")
    
    st.metric(
        "Podium Probability",
        f"{probability*100:.2f}%"
    )
    st.progress(float(probability))

st.markdown("---")
st.subheader("📈 Podium Probability vs Grid Position")

grid_positions = []
probabilities = []

for grid in range(1, 21):

    chart_input = pd.DataFrame({
        "GridPosition": [grid],
        "DriverEncoded": [
            driver_encoder.transform([driver])[0]
        ],
        "TeamEncoded": [
            team_encoder.transform([team])[0]
        ],
        "AvgDriverPoints": [avg_points],
        "AvgDriverFinish": [avg_finish]
    })

    prob = model.predict_proba(chart_input)[0][1]

    grid_positions.append(grid)
    probabilities.append(prob * 100)

chart_df = pd.DataFrame({
    "Grid Position": grid_positions,
    "Probability": probabilities
})

fig = px.line(
    chart_df,
    x="Grid Position",
    y="Probability",
    markers=True,
    title=f"{driver} Podium Probability by Grid Position"
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    title_x=0.5,
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(size=14),
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.update_traces(
    line=dict(width=4),
    marker=dict(size=8)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

with st.expander("📖 About This Project"):

    st.markdown("""
    ### Formula 1 Podium Predictor

    This dashboard predicts whether an F1 driver is likely to finish on the podium using machine learning.

    #### Dataset
    - 2025 Formula 1 Season
    - 479 Driver-Race Records

    #### Features
    - Grid Position
    - Driver Identity
    - Constructor Team
    - Average Driver Points
    - Average Driver Finish Position

    #### Models Evaluated
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

    #### Best Model
    Logistic Regression

    **F1 Score:** 0.786
    """)