import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="CardioVerse AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= LOAD MODEL =================

model = joblib.load("LR_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ================= GLOBAL CSS =================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
}

/* MAIN BACKGROUND */

.stApp{
    background:
    radial-gradient(circle at top left, rgba(255,0,128,0.18), transparent 25%),
    radial-gradient(circle at bottom right, rgba(0,212,255,0.18), transparent 25%),
    linear-gradient(135deg, #020617, #0f172a, #000000);

    color:white;
    overflow-x:hidden;
}

/* REMOVE STREAMLIT DEFAULT */

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

/* GLASS CARD */

.glass-card{

    background: rgba(255,255,255,0.05);

    border:1px solid rgba(255,255,255,0.08);

    border-radius:30px;

    padding:35px;

    backdrop-filter:blur(20px);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);

    transition:0.4s ease;
}

.glass-card:hover{

    transform:translateY(-5px);

    box-shadow:
        0 15px 40px rgba(255,0,100,0.18);
}

/* INPUT LABELS */

.stSelectbox label,
.stSlider label,
.stNumberInput label{

    color:white !important;

    font-weight:600;
}

/* INPUT FIELDS */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{

    background-color:rgba(255,255,255,0.05) !important;

    color:white !important;

    border-radius:15px !important;

    border:1px solid rgba(255,255,255,0.08) !important;
}

/* BUTTON */

.stButton > button{

    width:100%;

    height:75px;

    border:none;

    border-radius:22px;

    background: linear-gradient(
        135deg,
        #ff0055,
        #7c3aed,
        #00d4ff
    );

    color:white;

    font-size:24px;

    font-weight:800;

    letter-spacing:1px;

    transition:0.4s;

    box-shadow:
        0 0 30px rgba(255,0,90,0.35);
}

.stButton > button:hover{

    transform:scale(1.03);

    box-shadow:
        0 0 50px rgba(0,212,255,0.60);
}

/* METRIC */

[data-testid="metric-container"]{

    background:rgba(255,255,255,0.05);

    border-radius:20px;

    padding:20px;

    border:1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 0 20px rgba(255,255,255,0.05);
}

/* RESULT */

.result-good{

    background:
    linear-gradient(
        135deg,
        rgba(0,255,120,0.18),
        rgba(0,255,120,0.05)
    );

    border:2px solid #00ff99;

    border-radius:30px;

    padding:40px;

    text-align:center;

    font-size:42px;

    font-weight:900;

    color:#00ff99;

    margin-top:40px;

    box-shadow:
        0 0 40px rgba(0,255,120,0.30);

    animation:pulse 2s infinite;
}

.result-bad{

    background:
    linear-gradient(
        135deg,
        rgba(255,0,90,0.18),
        rgba(255,0,90,0.05)
    );

    border:2px solid #ff0055;

    border-radius:30px;

    padding:40px;

    text-align:center;

    font-size:42px;

    font-weight:900;

    color:#ff4d6d;

    margin-top:40px;

    box-shadow:
        0 0 40px rgba(255,0,90,0.30);

    animation:pulse 2s infinite;
}

@keyframes pulse{

    0%{
        transform:scale(1);
    }

    50%{
        transform:scale(1.01);
    }

    100%{
        transform:scale(1);
    }
}

/* FOOTER */

.footer{

    text-align:center;

    color:#64748b;

    margin-top:50px;

    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO SECTION =================

components.html("""
<!DOCTYPE html>
<html>
<head>
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

body{
    margin:0;
    padding:0;
    background:transparent;
    overflow:hidden;
    font-family:'Orbitron', sans-serif;
}

.hero-container{

    position:relative;

    padding:70px 50px;

    border-radius:35px;

    overflow:hidden;

    background: rgba(255,255,255,0.04);

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(20px);

    box-shadow:
        0 0 40px rgba(255,0,90,0.15),
        0 0 80px rgba(0,212,255,0.10);
}

/* GLOW */

.hero-container::before{
    content:'';

    position:absolute;

    width:600px;
    height:600px;

    background: radial-gradient(circle,
    rgba(255,0,90,0.25),
    transparent 70%);

    top:-250px;
    left:-200px;

    animation: float1 10s ease-in-out infinite;
}

.hero-container::after{
    content:'';

    position:absolute;

    width:600px;
    height:600px;

    background: radial-gradient(circle,
    rgba(0,212,255,0.20),
    transparent 70%);

    bottom:-250px;
    right:-200px;

    animation: float2 12s ease-in-out infinite;
}

@keyframes float1{
    0%{transform:translateY(0px);}
    50%{transform:translateY(30px);}
    100%{transform:translateY(0px);}
}

@keyframes float2{
    0%{transform:translateY(0px);}
    50%{transform:translateY(-30px);}
    100%{transform:translateY(0px);}
}

/* TITLE */

.hero-title{

    position:relative;

    z-index:2;

    text-align:center;

    font-size:85px;

    font-weight:900;

    letter-spacing:8px;

    background: linear-gradient(
        90deg,
        #ff0055,
        #ff4d9d,
        #00d4ff,
        #ffffff,
        #00d4ff,
        #ff0055
    );

    background-size:300% auto;

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    animation: shine 7s linear infinite;
}

@keyframes shine{
    to{
        background-position:300% center;
    }
}

/* SUBTITLE */

.hero-subtitle{

    position:relative;

    z-index:2;

    text-align:center;

    color:#cbd5e1;

    font-size:22px;

    margin-top:15px;
}

/* BADGE */

.hero-badge{

    position:relative;

    z-index:2;

    width:fit-content;

    margin:auto;

    margin-top:30px;

    padding:12px 28px;

    border-radius:999px;

    background:rgba(255,255,255,0.05);

    border:1px solid rgba(255,255,255,0.10);

    color:#00d4ff;

    font-weight:700;

    backdrop-filter:blur(15px);

    box-shadow:
        0 0 25px rgba(0,212,255,0.18);
}

</style>
</head>

<body>

<div class="hero-container">

    <div class="hero-title">
        CARDIOVERSE AI
    </div>

    <div class="hero-subtitle">
        Futuristic AI Powered Cardiac Intelligence System
    </div>

    <div class="hero-badge">
        ⚡ Neural Heart Risk Engine • Real-Time Analysis
    </div>

</div>

</body>
</html>
""", height=320)

# ================= MAIN LAYOUT =================

left, right = st.columns([2.2,1])

# ================= INPUT SECTION =================

with left:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("🧬 Patient Health Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 100, 40)

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            80, 200, 120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            100, 600, 200
        )

    with col2:

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0,1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.slider(
            "Max Heart Rate",
            60, 220, 150
        )

        exercise_angina = st.selectbox(
            "Exercise Angina",
            ["Y", "N"]
        )

        oldpeak = st.slider(
            "Oldpeak",
            0.0, 6.0, 1.0
        )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    predict_btn = st.button(
        "🚀 ANALYZE HEART RISK"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ================= STATS SECTION =================

with right:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("📊 Live Stats")

    st.metric("🧓 Age", age)

    st.metric("🩸 Blood Pressure", resting_bp)

    st.metric("🫀 Cholesterol", cholesterol)

    st.metric("⚡ Max Heart Rate", max_hr)

    st.progress(min(age / 100, 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION =================

if predict_btn:

    raw_input = {

        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,

        'ChestPainType_' + chest_pain: 1,

        'RestingECG_' + resting_ecg: 1,

        'ExerciseAngina_' + exercise_angina: 1,

        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:

        st.markdown("""
        <div class="result-bad">
            ⚠️ HIGH RISK OF HEART DISEASE
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-good">
            ✅ LOW RISK OF HEART DISEASE
        </div>
        """, unsafe_allow_html=True)

# ================= FOOTER =================

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit • Machine Learning • Futuristic UI
</div>
""", unsafe_allow_html=True)