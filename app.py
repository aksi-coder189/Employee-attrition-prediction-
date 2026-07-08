import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + DARK PREMIUM THEME
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG        = "#1a1a2e"
PANEL     = "#16213e"
BORDER    = "#0f3460"
ACCENT    = "#e94560"
TEAL      = "#64ffda"
TEXT      = "#ccd6f6"
MUTED     = "#a8b2d8"
AMBER     = "#f5a623"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": BORDER,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
})

st.markdown(f"""
<style>
    .stApp {{
        background: {BG};
    }}
    section[data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {BORDER};
    }}
    h1, h2, h3 {{ color: {ACCENT} !important; }}
    p, li, span, label {{ color: {TEXT}; }}
    .hero {{
        background: linear-gradient(135deg, {BG}, {BORDER});
        border-radius: 16px;
        padding: 32px 36px;
        border: 1px solid {ACCENT};
        margin-bottom: 20px;
    }}
    .card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }}
    .metric-box {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-left: 4px solid {ACCENT};
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }}
    .metric-box .val {{ color: {TEAL}; font-size: 1.6rem; font-weight: 700; }}
    .metric-box .lbl {{ color: {MUTED}; font-size: 0.85rem; }}
    .badge {{
        display:inline-block; background:{BORDER}; color:{TEAL};
        padding:4px 12px; border-radius:20px; font-size:0.8rem; margin-right:6px;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
    .stButton>button {{
        background: linear-gradient(135deg, {ACCENT}, {BORDER});
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


def metric_box(label, value):
    st.markdown(f"""
    <div class="metric-box">
        <div class="val">{value}</div>
        <div class="lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1 style="margin:0;">🧠 Employee Attrition Prediction</h1>
    <hr style="border-color:{BORDER}; margin:16px 0;">
    <p style="color:{TEXT}; line-height:1.7; margin:0;">
        This dashboard runs the full IBM HR Analytics attrition pipeline end-to-end:
        data exploration, preprocessing, EDA, model training &amp; comparison
        (Logistic Regression, Random Forest, Gradient Boosting), evaluation, and
        HR-ready business recommendations — all from your uploaded dataset.
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — DATA INPUT
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📂 Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload IBM HR Attrition CSV",
    type=["csv"],
    help="WA_Fn-UseC_-HR-Employee-Attrition.csv (or any dataset with the same schema)"
)
use_sample_note = st.sidebar.checkbox("I don't have the file — show me what's needed", value=False)

if use_sample_note:
    st.sidebar.info(
        "Download **WA_Fn-UseC_-HR-Employee-Attrition.csv** "
        "(IBM HR Analytics Employee Attrition & Performance dataset, Kaggle) "
        "and upload it above."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Model Settings")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
run_button = st.sidebar.button("🚀 Run Full Analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · pandas · seaborn")


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS (cached)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def preprocess(df):
    raw = df.copy()

    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    cat_features = df.select_dtypes(include='object').columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_features, drop_first=True)

    X = df_enc.drop('Attrition', axis=1)
    y = df_enc['Attrition']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return raw, X_scaled.reset_index(drop=True), y.reset_index(drop=True), cat_features


@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train, _random_state):
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=500, random_state=_random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=_random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, random_state=_random_state
        ),
    }

    trained = {}
    for name, model in models.items():
        if name == "Gradient Boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(trained, X_test, y_test):
    results = {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'y_pred': y_pred,
            'y_prob': y_prob,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.info("👈 Upload the IBM HR Attrition CSV from the sidebar, then click **Run Full Analysis**.")
    st.stop()

df = load_data(uploaded.getvalue())

required_cols = {'Attrition'}
if not required_cols.issubset(df.columns):
    st.error("This file doesn't look like the IBM HR Attrition dataset — an `Attrition` column is required.")
    st.stop()

if not run_button:
    st.success(f"File loaded: **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns. "
               f"Click **🚀 Run Full Analysis** in the sidebar to continue.")
    st.dataframe(df.head(10), use_container_width=True)
    st.stop()

with st.spinner("Running full pipeline — preprocessing, EDA, training, evaluation..."):
    raw, X_scaled, y, cat_features = preprocess(df)

    attrition_counts = raw['Attrition'].value_counts()
    attrition_rate = (attrition_counts.get('Yes', 0) / len(raw)) * 100

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    trained = train_models(X_train, y_train, random_state)
    results = evaluate_models(trained, X_test, y_test)

    metrics_df = pd.DataFrame({
        name: {k: round(v, 4) for k, v in vals.items() if k not in ('y_pred', 'y_prob')}
        for name, vals in results.items()
    }).T
    best_name = metrics_df['ROC-AUC'].astype(float).idxmax()
    best_model = trained[best_name]


# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════
tab_overview, tab_eda, tab_models, tab_eval, tab_insights = st.tabs(
    ["📋 Overview", "📊 EDA", "🤖 Models", "🎯 Evaluation", "💡 HR Insights"]
)

# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────
with tab_overview:
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_box("Rows", f"{raw.shape[0]:,}")
    with c2: metric_box("Columns", raw.shape[1])
    with c3: metric_box("Attrition Rate", f"{attrition_rate:.1f}%")
    with c4: metric_box("Class Imbalance", "Yes ⚠️" if attrition_rate < 30 else "No")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**First 10 rows**")
        st.dataframe(raw.head(10), use_container_width=True)
    with col_b:
        st.markdown("**Missing values**")
        miss = raw.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.success("No missing values found in the dataset ✅")
        else:
            st.dataframe(miss.to_frame("Missing Count"), use_container_width=True)

    st.markdown("**Statistical summary**")
    st.dataframe(raw.describe(), use_container_width=True)

    num_cols = raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.markdown(
        f'<span class="badge">🔢 Numeric: {len(num_cols)}</span>'
        f'<span class="badge">🔤 Categorical: {len(cat_features)}</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### ✅ Preprocessing Summary")
    st.markdown(f"""
    <div class="card">
    <ul>
        <li>Dropped uninformative columns: <code>EmployeeNumber, Over18, StandardHours, EmployeeCount</code></li>
        <li>Encoded target — <b style="color:{TEAL}">Attrition</b>: Yes → 1, No → 0</li>
        <li>One-Hot Encoded {len(cat_features)} categorical features</li>
        <li>Applied <code>StandardScaler</code> to all features</li>
        <li>Final feature matrix: <b style="color:{TEAL}">{X_scaled.shape[0]} rows × {X_scaled.shape[1]} features</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 2: EDA ────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis")

    raw_eda = raw.copy()
    raw_eda['AttritionNum'] = (raw_eda['Attrition'] == 'Yes').astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Employee Attrition — EDA Dashboard', fontsize=17, color=ACCENT, fontweight='bold', y=1.02)

    if 'Department' in raw_eda.columns:
        dep_rate = (raw_eda.groupby('Department')['AttritionNum'].mean() * 100).sort_values()
        axes[0, 0].barh(dep_rate.index, dep_rate.values, color=TEAL, edgecolor=BORDER)
        axes[0, 0].set_title('Attrition Rate by Department (%)', color=ACCENT)

    if 'OverTime' in raw_eda.columns:
        ot_rate = (raw_eda.groupby('OverTime')['AttritionNum'].mean() * 100)
        axes[0, 1].bar(ot_rate.index, ot_rate.values, color=[ACCENT, TEAL], edgecolor=BORDER)
        axes[0, 1].set_title('Attrition Rate by OverTime (%)', color=ACCENT)

    if 'WorkLifeBalance' in raw_eda.columns:
        wlb_rate = (raw_eda.groupby('WorkLifeBalance')['AttritionNum'].mean() * 100)
        axes[1, 0].bar(wlb_rate.index.astype(str), wlb_rate.values, color=AMBER, edgecolor=BORDER)
        axes[1, 0].set_title('Attrition Rate by Work-Life Balance (%)', color=ACCENT)

    if 'MonthlyIncome' in raw_eda.columns:
        sns.boxplot(data=raw_eda, x='Attrition', y='MonthlyIncome', ax=axes[1, 1],
                    palette={'No': TEAL, 'Yes': ACCENT})
        axes[1, 1].set_title('Monthly Income by Attrition', color=ACCENT)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    if 'JobRole' in raw_eda.columns:
        fig2, ax2 = plt.subplots(figsize=(13, 6))
        fig2.patch.set_facecolor(BG)
        role_rate = (raw_eda.groupby('JobRole')['AttritionNum'].mean() * 100).sort_values()
        colors = [ACCENT if v > 15 else TEAL for v in role_rate.values]
        ax2.barh(role_rate.index, role_rate.values, color=colors, edgecolor=BORDER)
        ax2.set_title('Attrition Rate by Job Role (%)', color=ACCENT, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.markdown(f"""
    <div class="card">
    <h4 style="margin-top:0;">💡 Key EDA Insights</h4>
    <p>Overall attrition rate in this dataset: <b style="color:{TEAL}">{attrition_rate:.1f}%</b>,
    confirming a class-imbalance problem that the models below correct for using balanced class weights.</p>
    <p>Watch for roles, low work-life balance ratings, and overtime status pushing attrition higher —
    and lower monthly income among employees who left.</p>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 3: MODELS ─────────────────────────────────────────────────────────
with tab_models:
    st.markdown("### 🤖 Model Training")
    st.markdown(f"""
    <div class="card">
    Three classifiers were trained on the preprocessed, scaled feature set with
    <b style="color:{TEAL}">balanced class weighting</b> to handle attrition's class imbalance:
    <ul>
        <li><b>Logistic Regression</b> — <code>class_weight='balanced'</code>, max_iter=500</li>
        <li><b>Random Forest</b> — 200 trees, <code>class_weight='balanced'</code></li>
        <li><b>Gradient Boosting</b> — 200 estimators, learning_rate=0.05, trained with balanced sample weights</li>
    </ul>
    Train/test split: <b style="color:{TEAL}">{int((1-test_size)*100)}% / {int(test_size*100)}%</b>,
    stratified on the target — {len(X_train)} training rows, {len(X_test)} test rows.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Model Performance Comparison")
    st.dataframe(
        metrics_df.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.4f}"),
        use_container_width=True
    )

    st.markdown(f"""
    <div class="metric-box" style="text-align:left; margin-top:10px;">
        <span style="color:{ACCENT}; font-weight:700;">🏆 Best Model: {best_name}</span><br>
        <span style="color:{MUTED};">Selected by highest ROC-AUC · ROC-AUC = {metrics_df.loc[best_name,'ROC-AUC']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 4: EVALUATION ─────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### 🎯 Confusion Matrices — All Models")
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.patch.set_facecolor(BG)
    fig3.suptitle('Confusion Matrices — All Models', color=ACCENT, fontsize=16, fontweight='bold')
    for ax, (name, vals) in zip(axes3, results.items()):
        cm = confusion_matrix(y_test, vals['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name, color=ACCENT, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.markdown("### Top 10 Feature Importances — Best Model")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
    feat_imp = pd.Series(importances, index=X_scaled.columns).sort_values(ascending=False)
    top10 = feat_imp.head(10)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.patch.set_facecolor(BG)
    colors_bar = [ACCENT if i == 0 else TEAL if i < 3 else MUTED for i in range(10)]
    ax4.barh(top10.index[::-1], top10.values[::-1], color=colors_bar[::-1], edgecolor=BORDER)
    ax4.set_title(f'Top 10 Feature Importances — {best_name}', color=ACCENT, fontsize=13)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)

    st.markdown("### ROC Curve — All Models")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    fig5.patch.set_facecolor(BG)
    colors_roc = [TEAL, ACCENT, AMBER]
    for (name, vals), color in zip(results.items(), colors_roc):
        fpr, tpr, _ = roc_curve(y_test, vals['y_prob'])
        ax5.plot(fpr, tpr, label=f"{name} (AUC={vals['ROC-AUC']:.3f})", color=color, linewidth=2.2)
    ax5.plot([0, 1], [0, 1], '--', color=MUTED, linewidth=1)
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve Comparison', color=ACCENT)
    ax5.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)

    st.markdown("### Sample Predictions (Best Model)")
    sample = X_test.iloc[:10]
    prediction = best_model.predict(sample)
    pred_df = pd.DataFrame({
        "Actual": y_test.iloc[:10].values,
        "Predicted": prediction,
        "Correct": ["✅" if a == p else "❌" for a, p in zip(y_test.iloc[:10].values, prediction)]
    })
    st.dataframe(pred_df, use_container_width=True)

# ── TAB 5: HR INSIGHTS ────────────────────────────────────────────────────
with tab_insights:
    top3 = top10.head(3)
    st.markdown(f"""
    <div class="card">
    <h3 style="margin-top:0;">🎯 Top 3 Predictors of Employee Exit</h3>
    <p>Based on the best-performing model (<b style="color:{TEAL}">{best_name}</b>), the strongest
    signals of attrition risk are:</p>
    <ol>
        <li><b style="color:{TEAL}">{top3.index[0]}</b> — importance score {top3.values[0]:.4f}</li>
        <li><b style="color:{TEAL}">{top3.index[1]}</b> — importance score {top3.values[1]:.4f}</li>
        <li><b style="color:{TEAL}">{top3.index[2]}</b> — importance score {top3.values[2]:.4f}</li>
    </ol>
    </div>

    <div class="card">
    <h3 style="margin-top:0;">🏁 Conclusion &amp; Business Recommendations</h3>
    <p><b>Key Findings</b></p>
    <ul>
        <li>Overall attrition rate: <b style="color:{TEAL}">{attrition_rate:.1f}%</b></li>
        <li>Employees flagged by overtime, low work-life balance, and lower income show elevated attrition risk</li>
        <li>The selected model (<b style="color:{TEAL}">{best_name}</b>) achieved the best balance of
            ROC-AUC ({metrics_df.loc[best_name,'ROC-AUC']:.4f}) and recall for identifying at-risk employees</li>
    </ul>
    <p><b>Recommendations</b></p>
    <ul>
        <li>Monitor employees working consistent overtime for burnout risk</li>
        <li>Prioritize retention initiatives for high-risk job roles and departments</li>
        <li>Review compensation bands for roles with below-median income and high attrition</li>
        <li>Use this model's risk scores to proactively flag employees for HR check-ins</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Model Comparison (CSV)",
        metrics_df.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )
