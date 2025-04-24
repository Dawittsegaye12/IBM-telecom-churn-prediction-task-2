import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load data
def load_data(filepath):
    data = pd.read_csv(r"C:\Users\dawit\telecom_churnPrediction\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna(subset=['TotalCharges'])
    data = pd.get_dummies(data, drop_first=True)
    return data

# Split data
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines and hyperparameter grids
def get_pipelines_and_params():
    log_reg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier())
    ])
    gb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier())
    ])

    log_reg_params = {
        'log_reg__C': [0.1, 1, 10],
        'log_reg__penalty': ['l2']
    }
    rf_params = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10]
    }
    gb_params = {
        'gb__n_estimators': [50, 100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'gb__max_depth': [3, 5, 10]
    }

    return {
        'log_reg': (log_reg_pipeline, log_reg_params),
        'rf': (rf_pipeline, rf_params),
        'gb': (gb_pipeline, gb_params)
    }

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    pipelines_and_params = get_pipelines_and_params()
    results = {}

    for model_name, (pipeline, params) in pipelines_and_params.items():
        random_search = RandomizedSearchCV(pipeline, params, cv=3, scoring='accuracy', n_iter=10, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        results[model_name] = {
            'best_params': random_search.best_params_,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

    return results

# Save the best model
def save_model(model, filepath):
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

# Update main function to save the best model dynamically
if __name__ == "__main__":
    filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(data, 'Churn_Yes')
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_model_name = None
    best_f1_score = 0
    best_model = None

    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        print(f"Best Parameters: {metrics['best_params']}\n")
        print(f"Accuracy: {metrics['accuracy']}\n")
        print(f"Precision: {metrics['precision']}\n")
        print(f"Recall: {metrics['recall']}\n")
        print(f"F1 Score: {metrics['f1_score']}\n")
        print(f"Classification Report:\n{metrics['classification_report']}\n")

        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_model_name = model_name
            best_model = random_search.best_estimator_  # Save the best model instance

    print(f"Best Model: {best_model_name} with F1 Score: {best_f1_score}")

    # Save the best model dynamically
    if best_model:
        save_model(random_search.best_estimator_, 'model/best_model.pkl')