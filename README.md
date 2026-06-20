# Car_Price_Prediction

An end-to-end Machine Learning web application built with Python, Scikit-Learn, and Flask that predicts the estimated resale value of a used car based on historical vehicle features, market data, and depreciation metrics.

The predictive core is powered by an optimized **Random Forest Regressor**, leveraging automated feature engineering and One-Hot categorical encoding to achieve a high $R^2$ accuracy score.

---

## 📊 System Architecture & Project Workflow

The application handles data processing and predictions seamlessly through the following pipeline:

1. **Data Ingestion & Cleaning**: Filters out unnecessary text scales and captures target-relevant numerical and categorical structures.
2. **Feature Engineering**: Optimizes learning metrics by transforming raw calendar metrics (`Year`) into explicit asset decay scales (`Car_Age = 2026 - Year`).
3. **Categorical Processing**: Vectorizes features like `Fuel_Type`, `Transmission`, and `Seller_Type` using dummy encoding flags (`pd.get_dummies(drop_first=True)`) to eliminate multi-collinearity.
4. **Model Serialization**: Freezes the optimized weights and configurations into a portable Python binary wrapper (`car_model.pkl`).
5. **Web Routing Deployment**: Integrates a Flask backend engine that accepts data parameters via a responsive UI form, reshapes inputs dynamically, and feeds them into the model for live predictions.

---

## 🛠️ Tech Stack & Dependencies

* **Core Language:** Python 3.x
* **Data Engineering:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn
* **Model Serialization:** Pickle Protocol
* **Backend Framework:** Flask (WSGI Web Server Gateway)
* **Frontend Design Layer:** Modern HTML5 & Glassmorphic CSS Grid

---

## 📁 Repository Directory Structure

Ensure your local environment matches the following directory layout:

```text
my_car_predictor/
│
├── app.py                # Core Flask backend server & routing logic
├── car_model.pkl         # Serialized high-performance Random Forest model
├── README.md             # Project implementation documentation
│
└── templates/
    └── index.html        # Mobile-responsive frontend prediction interface form
