# Insurance Premium Prediction API

This project is a FastAPI-based web service for predicting insurance premium categories using a machine learning model trained on user health and demographic data.

## Features

- REST API endpoint for premium prediction
- Input validation using Pydantic models
- Machine learning pipeline with scikit-learn
- Predicts premium category (Low, Medium, High) based on user data

## Tech Stack

- FastAPI
- Pydantic
- scikit-learn
- pandas

## How It Works

1. User submits data (age, weight, height, income, smoker status, city, occupation).
2. API computes derived features (BMI, age group, lifestyle risk, city tier).
3. Data is passed to a trained scikit-learn model.
4. API returns the predicted insurance premium category.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/insurance-premium-fastapi-ml.git
cd insurance-premium-fastapi-ml
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional)

If you want to retrain the model, use the provided Jupyter notebook (`fastapi_ml_model.ipynb`) and save the model as `model.pkl`.

### 5. Run the API server

```bash
uvicorn app:app --reload
```

### 6. Test the API

Send a POST request to `http://127.0.0.1:8000/predict` with JSON data:

```json
{
  "age": 35,
  "weight": 70.0,
  "height": 1.75,
  "income_lpa": 10.0,
  "smoker": false,
  "city": "Delhi",
  "occupation": "private_job"
}
```

## Files

- `app.py` — FastAPI application
- `insurance.csv` — Training data
- `model.pkl` — Trained ML model
- `fastapi_ml_model.ipynb` — Model training notebook

## License

MIT

---

**Feel free to adjust the project/repo name and README content as needed!**