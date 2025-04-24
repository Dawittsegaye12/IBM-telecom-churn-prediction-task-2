# Telecom Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecom company using machine learning models. The analysis provides actionable insights to reduce churn rates and improve customer retention. The project includes data preprocessing, exploratory data analysis (EDA), model development, and deployment of a prediction system.

## Project Structure
The project is organized as follows:

```
telecom_churnPrediction/
│
├── flask_backend.py          # Flask backend for serving predictions
├── model_development.py      # Model training and evaluation script
├── streamlit_frontend.py     # Streamlit app for user interaction
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── model/
│   └── best_model.pkl        # Trained model file
├── notebooks/
│   ├── eda.ipynb            # Exploratory Data Analysis notebook
│   └── requierements.txt     # Required Python packages
├── images/
│   └── Screenshot (183).png # Placeholder for screenshots
└── README.md                 # Project documentation
```

## Key Components

### 1. Dataset
- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Contains customer information, including demographic details, account information, and service usage.
- Target variable: `Churn_Yes` (1 = Churn, 0 = No Churn).

### 2. Model Development
- **File**: `model_development.py`
- Trains and evaluates machine learning models (Logistic Regression, Random Forest, Gradient Boosting).
- Uses `RandomizedSearchCV` for hyperparameter tuning.
- Saves the best model to `model/best_model.pkl`.

### 3. Backend
- **File**: `flask_backend.py`
- Flask API for serving predictions.
- Endpoints:
  - `/predict`: Accepts customer data and returns churn prediction.
  - `/batch_predict`: Accepts multiple customer records for batch predictions.

### 4. Frontend
- **File**: `streamlit_frontend.py`
- Streamlit app for user interaction.
- Allows users to input customer details and view predictions.

### 5. Exploratory Data Analysis (EDA)
- **File**: `notebooks/eda.ipynb`
- Analyzes data distribution, relationships, and key features.
- Includes visualizations like histograms, box plots, and correlation heatmaps.

## How to Run the Project

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install -r notebooks/requierements.txt
```

### 2. Run the Backend
Start the Flask backend:
```bash
python flask_backend.py
```

### 3. Run the Frontend
Start the Streamlit app:
```bash
streamlit run streamlit_frontend.py
```

### 4. Interact with the App
- Open the Streamlit app in your browser.
- Input customer details to get churn predictions.

## Screenshots
![App Screenshot](images/Screenshot%20(183).png)

## Future Improvements
- Add more features to the dataset for better predictions.
- Deploy the app on a cloud platform (e.g., AWS, Heroku).
- Implement advanced visualization techniques in the EDA.

## License
This project is licensed under the MIT License.