# 📦 Planning & Scheduling Assistant

A user-friendly, interactive dashboard and AI-powered assistant for exploring, predicting, and analyzing sales order delays.  
Built using **Streamlit** and **Gemini LLM** for rapid business insights and explainable ML.

Link to UI: https://planning-scheduling-assistant-r5tabnpwkeexvtinr63oyn.streamlit.app/
---

## What’s Inside?

- **EDA Dashboard:** Interactive, filterable insights about order delays, sales regions, product types, and more.
- **ML Model Builder:** Train and evaluate Random Forest or Logistic Regression models to predict delivery delays.
- **Instant Prediction:** Enter a new order and predict the probability of delay in real-time.
- **LLM Chat Assistant:** Ask questions in plain English and get data-driven answers with code and explanations, powered by Gemini (Google AI).
- **Fully modular codebase:** Clear separation of ingestion, EDA, modeling, UI, and LLM logic for maintainability and extensibility.

---

## 🗂️ Project Structure
```Planning-Scheduling-Assistant/
├── app.py # Main Streamlit app: dashboard, routing, and logic
├── introduction.py # Friendly UI intro & column dictionary
├── requirements.txt # All Python dependencies
├── .gitignore # Files/folders NOT tracked by git

├── data/
│ └── sales_data.csv # Dataset

├── src/
│ ├── ingest.py # DataIngestor: load, clean, feature engineering
│ ├── explore.py # DataExplorer: KPIs, plots, grouped analysis
│ ├── model.py # DelayPredictor: train/evaluate/predict
│ ├── ui.py # UIComponents: Streamlit controls/widgets
│ ├── llm.py # LLMAssistant: Gemini-powered chat/analysis


├── train_default_model.py # Script to train and save a default model
├── test_ingest.py # Test/demo files (optional)
├── test_gemini_api.py
└── README.md # This documentation!
```


---

## 📊 About the Data

The dataset contains **15,000+ sales order records** over five years, with the following fields:

- **Branch, Region, Zone:** The geographic sales segment handling the order.
- **Order, Shipment, Line Identifiers:** For tracking orders across business units.
- **Order Dates:** Creation, promised delivery, scheduled pickup, actual completion.
- **Product Information:** Product type, unit price (SRP1), sales rep.
- **Status Flags:** "URGENT" alerts, delivery status, ECD Notes (text reasons for delays).
- **Engineered Features:** 
    - `delay_flag`: Whether order was late
    - `delay_days`: How many days late
    - `lead_time`: Time from creation to delivery
    - `pickup_lead`: Creation to pickup time
    - `total_units`: Calculated from price and amount
    - `is_urgent`, `month`, `year`, `week`, etc.

*All columns and features are explained in detail in the app’s Introduction tab.*

---

## 🧑‍💻 How to Use This App

### 1. Clone the Repo

```bash
git clone https://github.com/saikirananugam/Planning-Scheduling-Assistant.git
cd Planning-Scheduling-Assistant
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Prepare Environment
The app needs a sample dataset at data/sales_data.csv (included for demo).

To use the LLM Assistant:
Add your Gemini API key as an environment variable (Gemini_API_KEY) or as a secret on Streamlit Cloud.

Locally: Create a .env file:
```bash
Gemini_API_KEY=sk-xxxxxxx
```
Streamlit Cloud: Use the “Secrets” panel to securely add your key.

### 4. Run Locally
```bash
streamlit run app.py
```
The app will open in your browser.
On first run, train a model in the “Predict Delivery Outcome” tab to enable ML predictions.

Explore Trends:
![image](https://github.com/user-attachments/assets/70cadfaa-4bd5-43f5-962e-5af53f299075)
![image](https://github.com/user-attachments/assets/e6c5c899-9182-4d3a-857d-514056208614)


Model Prediction:
![image](https://github.com/user-attachments/assets/36c33b50-bee7-4758-bdd6-a2a993dd5f50)

Using the Chatbot to ask the questions related to data:
![image](https://github.com/user-attachments/assets/bc6dd188-2c54-4b25-9fe9-c6f405fa3dce)





