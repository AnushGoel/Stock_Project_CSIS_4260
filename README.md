# 📈 CSIS 4260 – Assignment 1: Stock Price Analysis & Prediction Dashboard

## 📌 Overview

This project integrates **research, benchmarking, and coding** using a time-series dataset of **daily stock prices** for **505 S&P 500 companies** (2013-02-08 to 2018-02-07). The assignment is divided into three parts, analyzing performance at **1x, 10x, and 100x** data scales.

## 📊 Dataset

- **Source:** S&P 500 stock prices  
- **Size:** 619,040 rows (~29MB CSV)  
- **Formats:**  
  - **CSV:** Original format  
  - **Parquet:** Possible conversion with compression ([Apache Parquet](https://arrow.apache.org/docs/python/parquet.html))

---

## 🛠️ Project Parts

### 🔹 Part 1: Data Storage & Retrieval ([📂 View Part A](https://github.com/AnushGoel/Stock_Project_CSIS_4290/blob/main/PartA.ipynb))

**Goal:** Optimize data storage by choosing between **CSV** and **Parquet**.

**Tasks:**
- Compare **read/write performance** at **1x, 10x, and 100x** scales.
- Assess performance, usability, and scalability.
- Recommend the best format based on benchmarking.

---

### 🔹 Part 2: Data Analysis & Prediction Models ([📂 View Part B](https://github.com/AnushGoel/Stock_Project_CSIS_4260/blob/main/PartB.ipynb))

**Goal:** Compare **Pandas** vs. **Polars** for data analysis and prediction modeling.

**Tasks:**
- Add **4+ technical indicators** (see [Investopedia](https://www.investopedia.com/terms/t/technicalindicator.asp)).
- Benchmark **data processing speed** using Pandas vs. Polars.
- Implement **two ML models** to predict next-day closing prices.
- Use an **80-20 train-test split** for backtesting.

---

### 🔹 Part 3: Dashboard Development ([🌐 View Streamlit Dashboard](https://mainpy-9fhvfvvqepopz9oz9jebvm.streamlit.app/#9ed967a8))

**Goal:** Build an interactive **dashboard** to visualize benchmarking and prediction results.

**Tasks:**
- **Section A:** Display **storage and dataframe performance** across scales.
- **Section B:** Enable **company ticker selection** for dynamic stock prediction visualization.
- Research dashboard frameworks:  
  - [Streamlit](https://streamlit.io/)  
  - [Dash](https://plotly.com/dash/)  
  - [Reflex](https://reflex.dev/)  

---

## 🚀 Getting Started

### ✅ Prerequisites

Ensure you have **Python 3.x** and install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📂 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AnushGoel/Stock_Project_CSIS_4290.git
cd Stock_Project_CSIS_4290
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### **1️⃣ Data Storage & Benchmarking**
Run the notebook for **Part A**:
```bash
jupyter notebook PartA.ipynb
```

### **2️⃣ Data Analysis & Model Training**
Run the notebook for **Part B**:
```bash
jupyter notebook PartB.ipynb
```

### **3️⃣ Running the Dashboard**
Launch the **Streamlit dashboard**:
```bash
streamlit run dashboard.py
```
or
```bash
python dashboard.py
```

---

## 📁 Project Structure

```bash
Stock_Project_CSIS_4290/
├── data/                   # Dataset (CSV/Parquet)
├── PartA.ipynb             # Notebook for Part 1 (Storage & Retrieval)
├── PartB.ipynb             # Notebook for Part 2 (Data Analysis & Modeling)
├── dashboard.py            # Dashboard application file
├── requirements.txt        # Dependencies
└── README.md               # Project overview
```

---

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request for improvements.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📢 Acknowledgments

- **Course:** CSIS 4260  
- **Instructor:** Nikhil Bhardwaj 
- **Dataset:** Provided by the course  
