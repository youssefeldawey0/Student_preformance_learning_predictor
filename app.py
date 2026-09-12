import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title='NEXUS • Student AI', page_icon='✦', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap");
html,body,[class*="css"]{font-family:Inter,sans-serif} .stApp{background:radial-gradient(circle at 15% 10%,#312e81 0,transparent 30%),radial-gradient(circle at 85% 20%,#164e63 0,transparent 30%),#070a16;color:#f8fafc}.block-container{max-width:1250px;padding-top:2rem}.hero{padding:34px;border:1px solid rgba(255,255,255,.13);border-radius:28px;background:linear-gradient(135deg,rgba(124,58,237,.24),rgba(6,182,212,.13));box-shadow:0 20px 80px rgba(0,0,0,.35)}h1{font-family:'Space Grotesk';font-size:3.2rem!important}.eyebrow{letter-spacing:3px;color:#67e8f9;font-weight:800}.glass{background:rgba(15,23,42,.72);border:1px solid rgba(255,255,255,.1);padding:22px;border-radius:22px}.stButton>button{width:100%;border:0;border-radius:14px;padding:14px;font-weight:800;background:linear-gradient(90deg,#8b5cf6,#06b6d4);color:white}.stMetric{background:rgba(255,255,255,.04);padding:14px;border-radius:16px}.stNumberInput input{border-radius:12px!important}.sidebar .sidebar-content{background:#0b1020}
</style>''', unsafe_allow_html=True)

FEATURES = ['Study_Hours', 'Attendance_Percentage', 'Assignment_Score', 'Quiz_Score', 'Project_Score', 'Midterm_Exam']

# --- IMPORTANT ---
# student_model.pkl / scaler.pkl were trained on DOUBLE-scaled data:
# final_data_preprocessing.ipynb first applied a RobustScaler (fit on the raw
# training split) and saved that to the "preprocessed" CSVs, then
# ml_models.ipynb loaded those already-scaled CSVs and fit a second
# StandardScaler (scaler.pkl) on top before training the model.
#
# Feeding raw 0-100 scores straight into scaler.pkl (skipping the first
# RobustScaler step) is why the app always predicted ~100% Pass: the raw
# values look like extreme outliers to a scaler fit on robust-scaled data.
#
# These are the exact RobustScaler statistics (median / IQR) computed from
# the same train_test_split(test_size=0.20, random_state=42, stratify=y)
# used in final_data_preprocessing.ipynb, so this reproduces the original
# training pipeline exactly (verified to match the notebook's 93.2% test
# accuracy).
ROBUST_MEDIAN = pd.Series({
    'Study_Hours': 6.70,
    'Attendance_Percentage': 68.70,
    'Assignment_Score': 54.65,
    'Quiz_Score': 47.90,
    'Project_Score': 56.60,
    'Midterm_Exam': 53.10,
})
ROBUST_IQR = pd.Series({
    'Study_Hours': 5.500,
    'Attendance_Percentage': 12.700,
    'Assignment_Score': 17.200,
    'Quiz_Score': 18.325,
    'Project_Score': 17.000,
    'Midterm_Exam': 20.800,
})


@st.cache_resource
def load_assets():
    if not (os.path.exists('student_model.pkl') and os.path.exists('scaler.pkl')):
        return None, None
    return joblib.load('student_model.pkl'), joblib.load('scaler.pkl')


model, scaler = load_assets()


def predict(raw_values):
    """raw_values: 2D array-like of shape (1, 6) in FEATURES order, raw 0-100 scale."""
    raw_df = pd.DataFrame(raw_values, columns=FEATURES)
    robust_scaled = (raw_df - ROBUST_MEDIAN) / ROBUST_IQR  # step 1: same RobustScaler as training
    fully_scaled = scaler.transform(robust_scaled)          # step 2: saved StandardScaler
    prob = float(model.predict_proba(fully_scaled)[0][1])
    return prob


with st.sidebar:
    st.markdown('## ✦ NEXUS AI')
    page = st.radio('Navigate', ['Prediction Studio', 'About the Model'])
    st.caption('Student Performance Intelligence')
    st.divider()
    st.info('Model: Logistic Regression\n\nFeatures: 6 academic indicators')

if page == 'Prediction Studio':
    st.markdown('<div class="hero"><div class="eyebrow">NEXT-GENERATION STUDENT ANALYTICS</div><h1>Will this student pass? ✦</h1><p>Enter academic indicators and let the trained machine-learning model estimate the probability of passing.</p></div>', unsafe_allow_html=True)
    st.write('')
    left, right = st.columns([1.15, .85], gap='large')
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader('Academic Signal Console')
        c1, c2 = st.columns(2)
        with c1:
            study = st.number_input('Study Hours', 0.0, 100.0, 10.0, 0.5)
            assignment = st.number_input('Assignment Score', 0.0, 100.0, 70.0, 1.0)
            project = st.number_input('Project Score', 0.0, 100.0, 70.0, 1.0)
        with c2:
            attendance = st.number_input('Attendance (%)', 0.0, 100.0, 80.0, 1.0)
            quiz = st.number_input('Quiz Score', 0.0, 100.0, 70.0, 1.0)
            midterm = st.number_input('Midterm Exam', 0.0, 100.0, 70.0, 1.0)
        predict_clicked = st.button('✦ RUN AI PREDICTION')
        st.markdown('</div>', unsafe_allow_html=True)
    values = np.array([[study, attendance, assignment, quiz, project, midterm]])
    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader('Live Student Profile')
        st.metric('Academic Average', f'{np.mean(values[0, 2:]):.1f}/100')
        st.metric('Attendance', f'{attendance:.0f}%')
        st.metric('Study Input', f'{study:.1f} hrs')
        st.markdown('</div>', unsafe_allow_html=True)
    if predict_clicked:
        if model is None:
            st.error('Model files were not found. Run train_model.py first to create student_model.pkl and scaler.pkl.')
        else:
            prob = predict(values)
            label = 'PASS' if prob >= 0.5 else 'AT RISK'
            a, b = st.columns([1, 1])
            with a:
                st.markdown(f'<div class="hero"><div class="eyebrow">AI VERDICT</div><h1>{label}</h1><h2>{prob*100:.1f}% confidence</h2><p>This is a model prediction, not an academic guarantee.</p></div>', unsafe_allow_html=True)
            with b:
                fig = go.Figure(go.Indicator(mode='gauge+number', value=prob*100, number={'suffix': '%'}, title={'text': 'Probability of Passing'}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#22d3ee'}}))
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                st.plotly_chart(fig, use_container_width=True)
            st.subheader('Feature Radar')
            radar = list(values[0])
            radar.append(radar[0])
            names = FEATURES + [FEATURES[0]]
            fig = go.Figure(go.Scatterpolar(r=radar, theta=names, fill='toself'))
            fig.update_layout(polar={'radialaxis': {'visible': True, 'range': [0, 100]}}, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'}, height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown('<div class="hero"><div class="eyebrow">MODEL ARCHITECTURE</div><h1>Inside NEXUS AI</h1><p>A quick look at how a student\'s academic profile becomes a pass/fail prediction.</p></div>', unsafe_allow_html=True)
    st.write('')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('#### 🧮 Algorithm')
        st.write('Logistic Regression, trained with scikit-learn.')
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('#### 📊 Input Features')
        st.write('Study Hours, Attendance %, Assignment, Quiz, Project, and Midterm scores.')
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('#### 🎯 Test Accuracy')
        st.write('~93% on a held-out test split.')
        st.markdown('</div>', unsafe_allow_html=True)

    st.write('')
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('''#### How a prediction is made

1. **Input** — six academic indicators are collected, each on their natural 0–100 scale.
2. **Preprocessing** — the same two-step scaling used during training is applied, in order, so the live inputs match the distribution the model learned from.
3. **Inference** — the scaled features are passed to the trained Logistic Regression model, which outputs a probability of passing.
4. **Output** — that probability is converted into a `PASS` / `AT RISK` verdict with a confidence score.

#### Why this matters

Keeping the exact preprocessing pipeline in sync between training and inference is what makes the prediction trustworthy — any mismatch there (even a subtle one) can silently skew every prediction toward one outcome.''')
    st.markdown('</div>', unsafe_allow_html=True)

    st.write('')
    st.caption('⚠️ Predictions are estimates based on historical data and are not a guarantee of academic outcomes.')
