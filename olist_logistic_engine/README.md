# Olist Logistic Engine

Logistics delivery performance analysis tool for Brazilian e-commerce.  
Analyzes order delivery accuracy, delays, and estimated vs actual gap based on Olist dataset.

## Data Source

**Data is not mine.**  
Credit: [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) on Kaggle

Original data provided by Olist under CC BY-NC-SA 4.0 license.

## Disclaimer

**This tool is for educational purposes only.**

- Delivery estimates are based on historical data and may not reflect current shipping conditions
- This tool does not constitute operational advice
- The author assumes no responsibility for business decisions made using this tool

## Features

- Order delivery performance dashboard
- Estimated vs actual delivery gap analysis
- On-time rate calculation by region and seller
- Delay severity analysis (worst-case delays)
- Geographic visualization of delivery performance

## Key Metrics

| Metric | Value |
|---|---|
| Orders Arrived Early | 89% |
| Avg Gap (Estimate vs Actual) | 11.18 days |
| Orders Actually Late | 7.8% |
| Worst Single Delay | 188.98 days |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py