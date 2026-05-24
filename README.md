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
| **Financial Planner** | LQ45 stock simulation for wedding, KPR, and education goals | • Monte Carlo simulation (10,000 paths)<br>• Risk profiling (Conservative/Moderate/Aggressive)<br>• COVID-19 crisis weighting (3x sampling)<br>• Interactive Streamlit dashboard<br>• What-If slider & probability gauge<br>• Dividend impact analysis<br>• Foreign flow & OHLC explorer|
| **Olist Logistics** | Brazilian e-commerce delivery route analysis to identify infrastructure bottlenecks | Route performance classification of 412 routes into Fast,Normal,Slow,Critical, XGBoost risk prediction with 78 percent ROC AUC, Supply demand gap analysis across 27 Brazilian states, DBSCAN clustering validation with silhouette score 0.387, What if simulation covering 355 routes and 29,664 orders, Cost benefit analysis proving warehouse investment is not feasible with negative 247.6 percent ROI|
| **BNSP Certification** | Project for Associate Data Scientist certification | End-to-end data pipeline, Missing value imputation (median/mode), Feature engineering (discount_depth, price_ratio, is_on_season), Regression models (Linear Regression, Random Forest, XGBoost), Business recommendations, Exportable model (pickle) |

---

## Tech Stack

- **Languages:** Python
- **Frameworks:** Streamlit
- **Libraries:** Pandas, NumPy, Scikit-learn, PyDeck, Matplotlib
- **Tools:** Jupyter Notebook, Git, VS Code

---

```markdown
## Quick Start

```bash
# Clone and install
git clone https://github.com/Afrodithena/kinaya_project_analysis.git
cd kinaya_project_analysis
pip install -r requirements.txt

# Financial Planner (LQ45)
# 1. Run lq45_data_processing.ipynb in Colab
# 2. cd financial_planner && streamlit run app.py

# Olist Logistics
# 1. Run olist_data_processing.ipynb in Colab
# 2. Place .parquet files in olist_logistic_engine/data/
# 3. cd olist_logistic_engine && streamlit run app.py
```
---
### Another Project Coming Soon! 🚀
