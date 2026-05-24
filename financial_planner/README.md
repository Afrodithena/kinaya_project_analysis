# LQ45 Financial Planning Engine

Financial planning tool for Wedding Fund, KPR Down Payment, and Child Education goals.  
Based on LQ45 Indonesia stock data (2019-2025) with crisis-weighted bootstrap simulation.

## Data Source

**Stock price data is not mine.**  
Credit: [Dataset-Saham-IDX](https://github.com/wildangunawan/Dataset-Saham-IDX) by wildangunawan  
Original source: Indonesia Stock Exchange (IDX)

##  Disclaimer

**This tool is for educational purposes only.**

- Past performance does not guarantee future results
- The calculations and projections are based on historical data (2019-2025) and may not reflect current market conditions
- This tool does not constitute financial advice
- Always consult with a licensed financial advisor before making investment decisions
- The author assumes no responsibility for any financial losses incurred using this tool

## Features

- Goal-based simulation (Wedding, KPR, Education)
- Crisis-weighted Monte Carlo (10.000 paths, 3x COVID period weight)
- Stock data explorer with OHLC charts and foreign flow
- Dividend impact analysis and capital breakdown

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py