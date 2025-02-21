# Stock_Project_CSIS_4290
Douglas College CSIS-4290 Assignment 1

## CSIS 4260 – Assignment 1: Stock Price Analysis & Prediction
Overview
This project combines research, benchmarking, and coding using a time-series dataset of daily stock prices for 505 S&P 500 companies (2013-02-08 to 2018-02-07). We evaluate three data scales: 1x, 10x, and 100x.

Dataset
Content: Daily stock prices (619,040 rows; ~29MB in CSV)
Alternative: Convert to Parquet with compression (Apache Parquet)
Parts
1. Data Storage & Retrieval
Goal: Decide between CSV and Parquet.
Tasks:
Benchmark read/write performance at scales 1x, 10x, and 100x.
Evaluate performance, usability, and scalability.
2. Data Analysis & Prediction Models
Goal: Compare Pandas vs. Polars while enhancing and analyzing the dataset.
Tasks:
Compute and add at least four technical indicators (see Investopedia).
Benchmark data processing performance.
Develop two prediction models for next-day closing prices using an 80-20 train-test split.
3. Dashboard Creation
Goal: Build an interactive dashboard displaying benchmark results and prediction outcomes.
Tasks:
Section A: Visualize benchmarking results for storage and dataframe performance.
Section B: Display prediction models with dynamic company ticker selection.
Research dashboard libraries (e.g., Streamlit, Dash, Reflex).
