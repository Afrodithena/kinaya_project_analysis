## About the Author

**Kinaya Rafa**

I am an aspiring data analyst and undergraduate student passionate about turning complex data into actionable insights. This portfolio documents my journey as I learn and grow in data science. I'm still building my logic through real-world problems. I know I still have a lot to learn, and I see each project as a step forward.

**If you're interested in connecting or collaborating, let's talk!**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kinaya_Rafa-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kinaya-rafa/)


# Kinaya Project Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)

A collection of data science projects by Kinaya Rafa, featuring financial planning tools and logistics network analysis.

---

## Projects Overview

| Project | Description | Key Features |
|---------|-------------|--------------|
| **Financial Planner** | LQ45 stock simulation for wedding, KPR, and education goals | Monte Carlo simulation (10,000 paths), Risk profiling (Conservative/Moderate/Aggressive), COVID-19 crisis weighting, Interactive Streamlit dashboard, What-If slider, Probability gauge |
| **Olist Logistics** | Brazilian e-commerce delivery route visualization and warehouse optimization | Route performance classification (Fast/Normal/Slow/Critical), Animated Pydeck map with play/pause, K-Means & DBSCAN warehouse optimization, Route risk prediction (Random Forest/XGBoost), Time-decay priority scoring, Cost-benefit analysis (ROI, payback, sensitivity)|
| **BNSP Certification** | Project for Associate Data Scientist certification | End-to-end data pipeline, Missing value imputation (median/mode), Feature engineering (discount_depth, price_ratio, is_on_season), Regression models (Linear Regression, Random Forest, XGBoost), Business recommendations, Exportable model (pickle) |

---

## Tech Stack

- **Languages:** Python
- **Frameworks:** Streamlit
- **Libraries:** Pandas, NumPy, Scikit-learn, PyDeck, Matplotlib
- **Tools:** Jupyter Notebook, Git, VS Code

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Afrodithena/kinaya_project_analysis.git
cd kinaya_project_analysis

# 2. Install dependencies
pip install -r requirements.txt

# FINANCIAL PLANNER

# 3. Prepare Data
# Run 'lq45_data_processing.ipynb' in Google Colab first
# This will generate cleaned stock data for 40 LQ45 companies
# 4. Run Streamlit
cd financial_planner
streamlit run app.py

# OLIST LOGISTICS

# 3. Prepare Data
# Run 'olist_data_processing.ipynb' in Google Colab first
# This will generate all required .parquet files
# Place generated files into: olist_logistic_engine/data/
# 4. Run Streamlit 
cd olist_logistic_engine
streamlit run app.py
```
---
### Another Project Coming Soon! 🚀
