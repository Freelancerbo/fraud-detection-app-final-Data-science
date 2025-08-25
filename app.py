import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---- Load Model Safely ----
try:
    model = joblib.load('fraud_detection_model.pkl')
except Exception as e:
    st.error(f"❌ Model load nahi ho saka: {e}")
    st.info("Make sure 'fraud_detection_model.pkl' is in the same folder.")
    st.stop()

# ---- Page Config ----
st.set_page_config(page_title="🛡️ FraudGuard AI", page_icon="🛡️", layout="centered")

# ---- Animated Header ----
st.markdown("""
<style>
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.animated-by {
    font-family: 'Courier New', monospace;
    font-size: 18px;
    text-align: center;
    margin: 20px 0;
    animation: pulse 2s infinite, blink 1.5s infinite;
    color: #FF4081;
    font-weight: bold;
}
</style>
<div class="animated-by">✨ Built by Tafseer Haider ✨</div>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🛡️ AI-Powered Fraud Detection System")

# ---- Sidebar ----
st.sidebar.header("⚡ Quick Actions")

# Load Fraud Sample
if st.sidebar.button("🎯 Load Fraud Sample"):
    # Real fraud-like values (from actual dataset patterns)
    st.session_state.update({
        'v1': -2.6,
        'v2': 1.8,
        'v3': -1.5,
        'v4': 3.0,
        'v5': -2.0,
        'amount': 2500.0
    })
    st.rerun()

# Load Normal Sample
if st.sidebar.button("✅ Load Normal Sample"):
    st.session_state.update({
        'v1': 0.0,
        'v2': 0.1,
        'v3': -0.2,
        'v4': 0.3,
        'v5': 0.0,
        'amount': 50.0
    })
    st.rerun()

# Clear Inputs
if st.sidebar.button("🗑️ Clear Inputs"):
    for i in range(1, 6):
        st.session_state[f"v{i}"] = 0.0
    st.session_state.amount = 0.0
    st.rerun()

# ---- Input Fields ----
st.subheader("📋 Enter Transaction Details")
cols = st.columns(3)
for i in range(1, 6):
    key = f"v{i}"
    if key not in st.session_state:
        st.session_state[key] = 0.0
    with cols[(i-1) % 3]:
        st.number_input(f"V{i}", value=st.session_state[key], format="%.6f", step=0.01, key=key)

st.subheader("💰 Transaction Amount")
if "amount" not in st.session_state:
    st.session_state.amount = 0.0
st.number_input("", value=st.session_state.amount, format="%.2f", step=1.0, key="amount")

# ---- Predict Button ----
st.markdown("---")
if st.button("🔮 Predict Fraud"):
    # Collect input
    input_data = [st.session_state[f"v{i}"] for i in range(1, 6)]
    input_data.append(st.session_state.amount)
    input_array = np.array([input_data])  # Shape: (1, 6)

    # Debug: Show what is being predicted
    with st.expander("🔍 Debug Info", expanded=False):
        st.write("Input Array:", input_array)
        st.write("Model Class List:", model.classes_)  # Should be [0 1]

    # Predict
    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]

    # Show result
    st.markdown("### 📊 Prediction Result")
    if pred == 1:
        st.error("🚨 **FRAUD DETECTED!**", icon="🚨")
    else:
        st.success("✅ **LEGITIMATE TRANSACTION**", icon="✅")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 2.5))
    classes = ['Not Fraud', 'Fraud']
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(classes, prob, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    for bar, p in zip(bars, prob):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{p:.4f}", ha='center')
    st.pyplot(fig)

    st.write(f"🔢 **Not Fraud:** {prob[0]:.4f} | **Fraud:** {prob[1]:.4f}")

else:
    st.info("👉 Use 'Load Fraud Sample' or enter values to test.")
