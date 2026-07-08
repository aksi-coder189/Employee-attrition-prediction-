# 🧠 Employee Attrition Prediction — Streamlit Dashboard

An interactive, dark-premium Streamlit dashboard that runs the full IBM HR Analytics Employee Attrition pipeline end-to-end — from raw CSV to trained models to HR-ready business recommendations.

Built from `analysis1.ipynb`, converted into a production-style web app so results can be explored interactively instead of scrolling through a notebook.

---

## 📌 Project Overview

| | |
|---|---|
| **Problem type** | Binary classification |
| **Target variable** | `Attrition` (Yes / No) |
| **Dataset** | IBM HR Analytics Employee Attrition & Performance (Kaggle) |
| **Models compared** | Logistic Regression, Random Forest, Gradient Boosting |
| **Selection metric** | ROC-AUC (with Recall as tiebreaker — catching actual leavers matters most in HR) |
| **Output** | Interactive dashboard: EDA, model comparison, evaluation charts, business insights |

**Business objective:** predict which employees are at risk of leaving so HR can intervene proactively — through targeted retention efforts, compensation review, or workload adjustments — before attrition happens.

---

## ✨ Features

- **📂 Upload-and-run** — no hardcoded dataset; upload any CSV with the same schema and the whole pipeline re-runs on it
- **📋 Overview tab** — shape, dtypes, missing values, statistical summary, preprocessing log
- **📊 EDA tab** — attrition rate by department, overtime, work-life balance, job role, and income distribution, all in the notebook's dark theme
- **🤖 Models tab** — trains all 3 classifiers live with balanced class weighting (handles the ~16% attrition class imbalance) and shows a colour-graded comparison table
- **🎯 Evaluation tab** — confusion matrices for all models, top-10 feature importances for the best model, ROC curve overlay, and sample predictions with correctness flags
- **💡 HR Insights tab** — auto-generated top-3 attrition predictors and business recommendations, plus a downloadable CSV of the model comparison table
- **⚙️ Adjustable settings** — test split size and random state configurable from the sidebar
- **⚡ Cached pipeline** — data loading, preprocessing, and model training are cached (`st.cache_data` / `st.cache_resource`) so re-visiting tabs is instant

---

## 🗂️ Project Structure

```
attrition_app/
├── app.py              # Main Streamlit application (single file)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧰 Requirements

- Python 3.9+
- pip

Dependencies (pinned in `requirements.txt`):

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## 🚀 Setup & Run

### 1. Get the project files
Place `app.py`, `requirements.txt`, and this `README.md` in a folder, e.g. `attrition_app/`.

### 2. Create a virtual environment (recommended)

**Windows**
```bash
cd attrition_app
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
cd attrition_app
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Streamlit will start a local server and open your browser automatically at:

```
http://localhost:8501
```

If it doesn't open automatically, copy that URL into your browser.

### 5. Stop the app
Press `Ctrl + C` in the terminal where Streamlit is running.

---

## 📖 How to Use

1. **Get the dataset.** Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` (IBM HR Analytics Employee Attrition & Performance, available on Kaggle). If you don't have it, tick **"I don't have the file"** in the sidebar for a pointer.
2. **Upload it.** Use the sidebar file uploader — the app validates that an `Attrition` column exists before proceeding.
3. **(Optional) Adjust settings.** Test-set size (default 20%) and random state (default 42) are adjustable in the sidebar.
4. **Click "🚀 Run Full Analysis."** This triggers preprocessing, EDA rendering, model training, and evaluation — the first run takes a few seconds; switching tabs afterward is instant due to caching.
5. **Explore the tabs:**
   - **Overview** → understand the raw data
   - **EDA** → see what's driving attrition visually
   - **Models** → compare Accuracy / Precision / Recall / F1 / ROC-AUC across all 3 models
   - **Evaluation** → confusion matrices, feature importance, ROC curves, sample predictions
   - **HR Insights** → plain-language findings and recommendations, downloadable as CSV
6. **Upload a different file anytime** — the app re-runs the full pipeline on it (any CSV with an `Attrition` Yes/No column and similar HR features will work, not just the original dataset).

---

## 🧪 Pipeline Details (what happens under the hood)

**Preprocessing**
- Drops uninformative columns: `EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount`
- Encodes target: `Attrition` → Yes = 1, No = 0
- One-Hot Encodes all categorical features (`drop_first=True`)
- Applies `StandardScaler` to the full feature matrix

**Modeling**
- Stratified train/test split (default 80/20) so class balance is preserved in both sets
- **Logistic Regression** — `class_weight='balanced'`, `max_iter=500`
- **Random Forest** — 200 trees, `class_weight='balanced'`
- **Gradient Boosting** — 200 estimators, `learning_rate=0.05`, trained with `sample_weight` from `compute_sample_weight('balanced', ...)` since `GradientBoostingClassifier` doesn't accept `class_weight` directly

**Evaluation**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC computed for all 3 models
- Best model selected by highest ROC-AUC
- Feature importances (or `abs(coef_)` for linear models) shown for the winning model

---

## 🖌️ Theme

Matches the notebook's dark-premium visual identity throughout the app and all matplotlib charts:

| Color | Hex | Use |
|---|---|---|
| Background | `#1a1a2e` | App background, figure background |
| Panel | `#16213e` | Cards, axes background |
| Border | `#0f3460` | Card borders, gridlines |
| Accent (red) | `#e94560` | Headings, highlights, primary bars |
| Teal | `#64ffda` | Positive highlights, secondary bars |
| Text | `#ccd6f6` | Body text |
| Muted | `#a8b2d8` | Labels, secondary text |
| Amber | `#f5a623` | Tertiary chart color |

---

## ❓ Troubleshooting

| Issue | Fix |
|---|---|
| `streamlit: command not found` | Activate your virtual environment first, or run `python -m streamlit run app.py` |
| "An `Attrition` column is required" error | Your CSV needs a column literally named `Attrition` with `Yes`/`No` values |
| App is slow on first run | Normal — training happens once; cached afterward. Re-uploading a file or changing test size/random state re-triggers training |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Charts look cut off / squished | Widen your browser window — the layout is set to `wide` mode |

---

## 📎 Notes

- The original dataset file isn't bundled with this project (it wasn't part of the source notebook) — upload it fresh each session via the sidebar.
- Any CSV sharing the same schema (an `Attrition` Yes/No column plus similar HR features) will run through the pipeline correctly.
- All computation happens locally in your Python environment — no data leaves your machine.

---

**Built for:** Employee Attrition Prediction
**Tech stack:** Streamlit · pandas · NumPy · scikit-learn · Matplotlib · Seaborn
