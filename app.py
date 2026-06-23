"""
💻 Laptop Price Predictor — Streamlit Web Application
Built by Divyansh Kushwaha | IIT Madras BS Data Science
"""

import streamlit as st
import pickle
import numpy as np

# ──────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CUSTOM CSS — dark-theme cards, gradient buttons, spacing
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
}

/* ── Hero header ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #8b949e;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* ── Section headings ── */
.section-heading {
    color: #58a6ff;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #21262d;
}

/* ── Input cards ── */
.input-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.input-card:hover {
    border-color: #388bfd;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #238636, #2ea043);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.85rem 2rem;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    margin-top: 1.5rem;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0);
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0d2137, #0f2d0e);
    border: 1px solid #238636;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 0 30px rgba(35, 134, 54, 0.15);
}
.result-label {
    color: #8b949e;
    font-size: 0.95rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-price {
    font-size: 3rem;
    font-weight: 700;
    color: #3fb950;
    line-height: 1.1;
}
.result-note {
    color: #8b949e;
    font-size: 0.82rem;
    margin-top: 0.6rem;
}

/* ── Error card ── */
.error-card {
    background: #1a0a0a;
    border: 1px solid #da3633;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    color: #ff7b72;
    margin-top: 1.2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
.sidebar-stat {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.7rem;
    color: #c9d1d9;
    font-size: 0.9rem;
}
.sidebar-stat span {
    color: #58a6ff;
    font-weight: 600;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #484f58;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #21262d;
    margin-top: 3rem;
}
.footer a { color: #58a6ff; text-decoration: none; }

/* ── Selectbox & slider labels ── */
label[data-testid="stWidgetLabel"] {
    color: #c9d1d9 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """Load the trained pipeline and reference dataframe from disk."""
    with open("pipe.pkl", "rb") as f:
        pipe = pickle.load(f)
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    return pipe, df

try:
    pipe, df = load_artifacts()
    model_loaded = True
except FileNotFoundError as exc:
    st.error(f"⚠️ Model file not found: {exc}. Make sure **pipe.pkl** and **df.pkl** are in the same directory as app.py.")
    model_loaded = False
    st.stop()
except Exception as exc:
    st.error(f"⚠️ Failed to load model: {exc}")
    model_loaded = False
    st.stop()


# ──────────────────────────────────────────────────────────
# SIDEBAR — project information & dataset stats
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💻 About This App")
    st.markdown("""
    This app uses a **Machine Learning ensemble model** trained on real-world laptop
    specifications to predict prices in Indian Rupees.
    """)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")

    total_laptops = len(df)
    brands = df["Company"].nunique()
    avg_price = int(df["Price"].mean())
    max_price = int(df["Price"].max())
    min_price = int(df["Price"].min())

    st.markdown(f'<div class="sidebar-stat">🗂️ Total laptops: <span>{total_laptops:,}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">🏷️ Brands covered: <span>{brands}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">💰 Avg price: <span>₹{avg_price:,}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">📈 Max price: <span>₹{max_price:,}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">📉 Min price: <span>₹{min_price:,}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown(f'<div class="sidebar-stat">🧠 Algorithm: <span>Voting Regressor</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">🎯 Base models: <span>RF · GBDT · XGB · ET</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">🔢 Features: <span>12 engineered</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat">📐 Target: <span>log(Price) → exp</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Fill in all laptop specifications using the inputs on the right.
    2. Click **🔍 Predict Price**.
    3. The predicted price in ₹ will appear instantly.
    """)


# ──────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">💻 Laptop Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Predict the price of any laptop using machine learning — powered by a Voting Regressor ensemble.</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# DROPDOWN / SLIDER OPTION LISTS (derived from training data)
# ──────────────────────────────────────────────────────────
company_list    = sorted(df["Company"].unique())
type_list       = sorted(df["TypeName"].unique())
ram_list        = sorted(df["Ram"].unique())
cpu_list        = sorted(df["Cpu brand"].unique())
gpu_list        = sorted(df["Gpu brand"].unique())
os_list         = sorted(df["os"].unique())

hdd_options     = [0, 32, 128, 500, 1000, 2000]
ssd_options     = [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024]

resolution_options = [
    "1920x1080", "1366x768", "1600x900", "3840x2160",
    "3200x1800", "2560x1440", "2560x1600", "1440x900", "2304x1440"
]


# ──────────────────────────────────────────────────────────
# INPUT FORM — three columns for compact layout
# ──────────────────────────────────────────────────────────
st.markdown('<p class="section-heading">⚙️ Laptop Specifications</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

# ── Column 1: Brand & type info ──
with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    company = st.selectbox(
        "🏷️ Brand",
        options=company_list,
        help="Laptop manufacturer / brand."
    )
    laptop_type = st.selectbox(
        "💼 Laptop Type",
        options=type_list,
        help="Category of the laptop."
    )
    ram = st.selectbox(
        "🧠 RAM (GB)",
        options=ram_list,
        index=ram_list.index(8),
        help="Amount of RAM in gigabytes."
    )
    weight = st.slider(
        "⚖️ Weight (kg)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.1,
        help="Laptop weight in kilograms."
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Column 2: Display info ──
with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    touchscreen = st.selectbox(
        "👆 Touchscreen",
        options=["No", "Yes"],
        help="Does the laptop have a touchscreen?"
    )
    ips = st.selectbox(
        "🖥️ IPS Display",
        options=["No", "Yes"],
        help="Does the laptop feature an IPS panel?"
    )
    screen_size = st.slider(
        "📏 Screen Size (inches)",
        min_value=10.0, max_value=18.0, value=15.6, step=0.1,
        help="Diagonal screen size in inches."
    )
    resolution = st.selectbox(
        "🔲 Screen Resolution",
        options=resolution_options,
        help="Native display resolution (width × height)."
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Column 3: Hardware ──
with col3:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    cpu_brand = st.selectbox(
        "⚡ CPU Brand",
        options=cpu_list,
        index=cpu_list.index("Intel Core i5"),
        help="Processor brand / tier."
    )
    hdd = st.select_slider(
        "💾 HDD Storage (GB)",
        options=hdd_options,
        value=0,
        help="Hard disk drive capacity in GB."
    )
    ssd = st.select_slider(
        "⚡ SSD Storage (GB)",
        options=ssd_options,
        value=256,
        help="Solid state drive capacity in GB."
    )
    gpu_brand = st.selectbox(
        "🎮 GPU Brand",
        options=gpu_list,
        help="Graphics card brand."
    )
    operating_system = st.selectbox(
        "🖥️ Operating System",
        options=os_list,
        help="Pre-installed operating system."
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors notebook preprocessing)
# ──────────────────────────────────────────────────────────
def calculate_ppi(resolution_str: str, inches: float) -> float:
    """
    Calculate pixels-per-inch (PPI) from a 'WxH' resolution string
    and the diagonal screen size in inches.
    Formula: sqrt(W² + H²) / diagonal_inches
    """
    try:
        x_res, y_res = map(int, resolution_str.lower().split("x"))
        ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / inches
        return round(ppi, 4)
    except Exception:
        return 141.0  # sensible default (1920×1080 @ ~15.6″)


import pandas as pd

def build_input_array(
    company, laptop_type, ram, weight,
    touchscreen, ips, screen_size, resolution,
    cpu_brand, hdd, ssd, gpu_brand, operating_system
):

    ts_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0
    ppi = calculate_ppi(resolution, screen_size)

    return pd.DataFrame({
        "Company": [company],
        "TypeName": [laptop_type],
        "Ram": [ram],
        "Weight": [float(weight)],
        "Touchscreen": [ts_val],
        "Ips": [ips_val],
        "ppi": [ppi],
        "Cpu brand": [cpu_brand],
        "HDD": [hdd],
        "SSD": [ssd],
        "Gpu brand": [gpu_brand],
        "os": [operating_system]
    })


# ──────────────────────────────────────────────────────────
# PREDICTION BUTTON & RESULT DISPLAY
# ──────────────────────────────────────────────────────────
st.markdown('<p class="section-heading">🔍 Predict</p>', unsafe_allow_html=True)

predict_clicked = st.button("🔍 Predict Price", use_container_width=True)

if predict_clicked:
    # Show a live PPI calculation for transparency
    ppi_display = calculate_ppi(resolution, screen_size)

    with st.spinner("Running the model…"):
        try:
            X = build_input_array(
                company, laptop_type, ram, weight,
                touchscreen, ips, screen_size, resolution,
                cpu_brand, hdd, ssd, gpu_brand, operating_system
            )

            # Model was trained on np.log(Price), so exponentiate the output
            log_price = pipe.predict(X)[0]
            predicted_price = np.exp(log_price)

            # ── Success card ──
            formatted = f"₹{predicted_price:,.2f}"
            st.markdown(f"""
            <div class="result-card">
                <p class="result-label">💰 Estimated Laptop Price</p>
                <p class="result-price">{formatted}</p>
                <p class="result-note">
                    Calculated PPI: <strong>{ppi_display:.1f}</strong> &nbsp;|&nbsp;
                    Resolution: <strong>{resolution}</strong> &nbsp;|&nbsp;
                    Screen: <strong>{screen_size}"</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Expandable debug info
            with st.expander("🔬 Feature vector sent to model"):
                import pandas as pd
                feature_names = [
                    "Company", "TypeName", "Ram (GB)", "Weight (kg)",
                    "Touchscreen", "IPS", "PPI",
                    "CPU Brand", "HDD (GB)", "SSD (GB)",
                    "GPU Brand", "OS"
                ]
                feature_values = X.iloc[0].tolist()
                debug_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Value": feature_values
                })
                st.dataframe(debug_df, use_container_width=True, hide_index=True)

        except Exception as e:
            import traceback
            st.code(traceback.format_exc())
            st.markdown(f'<div class="error-card">⚠️ Prediction failed: <code>{e}</code></div>',
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <strong>Divyansh Kushwaha</strong> | IIT Madras BS Data Science &nbsp;·&nbsp;
    Powered by <a href="https://streamlit.io" target="_blank">Streamlit</a> &amp;
    <a href="https://scikit-learn.org" target="_blank">scikit-learn</a>
</div>
""", unsafe_allow_html=True)