# Telecom Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn for a telecom company using machine learning models. The goal is to identify customers likely to churn and provide actionable insights to improve customer retention strategies. The project includes data preprocessing, exploratory data analysis (EDA), model development, and deployment of a prediction system.

## Objectives
- Predict customer churn using machine learning models.
- Analyze key factors contributing to churn.
- Provide a user-friendly interface for predictions.
- Deploy a scalable backend for serving predictions.

## Dataset
- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Description**: The dataset contains customer information, including demographic details, account information, and service usage.
- **Target Variable**: `Churn_Yes` (1 = Churn, 0 = No Churn).
- **Key Features**:
  - `tenure`: Number of months the customer has been with the company.
  - `MonthlyCharges`: Monthly charges for the customer.
  - `TotalCharges`: Total charges incurred by the customer.
  - `Contract`, `PaymentMethod`, and other categorical features.

## Methodology

### 1. Data Preprocessing
- Handled missing values in the `TotalCharges` column by converting it to numeric and dropping rows with missing values.
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features for model compatibility.

### 2. Exploratory Data Analysis (EDA)
- Analyzed data distribution using histograms and box plots.
- Identified correlations between features and the target variable using a heatmap.
- Checked for class imbalance in the target variable.

### 3. Model Development
- Trained and evaluated three machine learning models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Used `RandomizedSearchCV` for hyperparameter tuning.
- Selected the best model based on F1-score.

### 4. Deployment
- Developed a Flask backend to serve predictions via API endpoints.
- Created a Streamlit frontend for user interaction.

## Results and Insights
- **Best Model**: Gradient Boosting (or dynamically selected based on F1-score).
- **Key Drivers of Churn**:
  - Short tenure
  - High monthly charges
  - Month-to-month contracts
- **Model Performance**:
  - Accuracy: ~85%
  - Precision: ~80%
  - Recall: ~75%
  - F1-Score: ~77%

## Challenges Faced
1. **Class Imbalance**:
   - The dataset had an imbalance in the target variable, with fewer customers labeled as churned. This was addressed using evaluation metrics like F1-score to balance precision and recall.

2. **Feature Engineering**:
   - Handling categorical variables and scaling numerical features required careful preprocessing to ensure compatibility with machine learning models.

3. **Execution Time**:
   - Hyperparameter tuning with `RandomizedSearchCV` was computationally expensive. This was mitigated by limiting the number of iterations and enabling parallel processing.

## Project Structure
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
- Integrate additional machine learning models for comparison.
- Automate hyperparameter tuning using tools like Optuna or Hyperopt.

## Conclusion
This project demonstrates the end-to-end process of building a machine learning system, from data preprocessing and model development to deployment. The insights derived from the analysis can help telecom companies identify at-risk customers and take proactive measures to improve retention. The deployed system provides a scalable solution for real-time churn prediction.

## License
This project is licensed under the MIT License.

## Authors
- **Dawit**: Data Scientist and Developer

## Acknowledgments
- Dataset provided by IBM Sample Data Sets.
- Inspiration from real-world telecom churn prediction challenges.