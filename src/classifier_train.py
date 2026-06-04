"""
Static gesture classifier training module.

Responsibilities:
- Load landmark dataset
- Preprocess training data
- Train machine learning classifier
- Evaluate model performance
- Save trained model

Current Planned Models:
- Random Forest
- SVM
- MLP

This module represents the first machine learning
training stage of the project.

Future Improvements:
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Advanced metrics
"""

import os 
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Load dataset
CSV_PATH = os.path.join("data", "landmarks", "landmark_dataset.csv")
MODEL_PATH = os.path.join("models", "trained", "gesture_classifier.pk1")

def main():
    print("Loading dataset...")

    df = pd.read_csv(CSV_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(df["label"].value_counts())

    X = df.drop(columns=["label", "timestamp", "hand_index"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        min_samples_split=5,
        min_samples_leaf=2,
        max_depth=10
        max_features="sqrt"
        max_samples=0.8,
        max_leaf_nodes=50,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluatig model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
