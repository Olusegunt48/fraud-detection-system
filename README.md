# Credit Card Fraud Detection System

A machine learning project that detects fraudulent credit card transactions in real time. The system takes in transaction data, analyses it using a trained model, and tells you whether a transaction is legitimate or fraudulent along with a probability score.

---

## What Is Inside

```
fraud_detection_system/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and split data
├── notebooks/
│   ├── 01_eda.ipynb            # Data exploration
│   ├── 02_preprocessing.ipynb  # Data preparation
│   └── 03_model_training.ipynb # Model training and evaluation
├── src/
│   ├── preprocess.py           # Preprocessing functions
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── api/
│   └── main.py                 # FastAPI backend
├── app/
│   └── streamlit_app.py        # Streamlit frontend
├── models/                     # Saved models
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place `creditcard.csv` inside `data/raw/`.

Dataset link: **https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**

The dataset contains 284,807 transactions of which only 492 are fraudulent. That is 0.17% fraud, which makes this a highly imbalanced classification problem.

---

## Getting Started

Clone the repository and set up your environment:

```bash
git clone https://github.com/Olusegunt48/fraud-detection-system.git
cd fraud-detection-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Run the three notebooks in order to generate your processed data and trained models:

```
01_eda.ipynb
02_preprocessing.ipynb
03_model_training.ipynb
```

---

## Running the App

You need two terminals open at the same time.

**Terminal 1:**
```bash
uvicorn api.main:app --reload
```

**Terminal 2:**
```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501` to use the app.

---

## Running With Docker

```bash
docker-compose up --build
```

FastAPI runs on port 8000 and Streamlit on port 8501. To stop:

```bash
docker-compose down
```

---

## How to Use the App

**Single Transaction** — Enter transaction values manually in the sidebar and click Analyse Transaction to get an instant prediction.

**Batch Upload** — Upload a CSV file with multiple transactions. The app analyses each one and returns a colour coded results table you can download.

---

## Model Performance

Three models were trained and compared. XGBoost was selected for deployment.

| Model | Recall (Fraud) | F1 (Fraud) | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.11 | 0.9708 |
| Random Forest | 0.83 | 0.85 | 0.9685 |
| XGBoost | 0.85 | 0.78 | 0.9800 |

---

## Built With

Python, Scikit learn, XGBoost, FastAPI, Streamlit, Docker

---

## Author

**Oluwasegun** — Senior BI Analyst and Data Scientist, Lagos, Nigeria