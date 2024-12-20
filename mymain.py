# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os
import pickle

# Elastic Net parameters for logistic regression
param_grid = {
    'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # Elastic Net mixing parameter
    'C': [0.01, 0.1, 1, 10]            # Regularization strength
}

# Initialize Logistic Regression with Elastic Net
def init_model():
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=1000,
        random_state=42
    )

# Function to train the model and make predictions
def train_and_predict():
    print("### Loading data... ###")
    # Load train and test data from the current directory
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Separate features and target
    X_train = train_data.drop(columns=["id", "sentiment", "review"])
    y_train = train_data["sentiment"]
    X_test = test_data.drop(columns=["id", "review"])
    ids = test_data["id"]

    print("\n### Performing Grid Search with Elastic Net Logistic Regression... ###")
    model = init_model()
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    # Save the trained model
    with open("best_model_vectorizer.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)
    print("Trained model saved to 'best_model.pkl'")

    # Predict probabilities on the test data
    print("\n### Generating Predictions on Test Data... ###")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Save predictions to CSV
    submission = pd.DataFrame({
        'id': ids,
        'prob': y_pred_proba
    })
    submission.to_csv("mysubmission.csv", index=False)
    print("Predictions saved to 'mysubmission.csv'")

# Main function
def main():
    print("\n### Sentiment Classification Script ###")
    train_and_predict()
    print("\n### Script Execution Complete! ###")

if __name__ == "__main__":
    main()
