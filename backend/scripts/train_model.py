import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(data_file="backend/datasets/combined_dataset.csv", model_file="backend/models/hand_gesture_model.pkl"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
 
    # Load the dataset
    data = pd.read_csv(data_file)
    
    # Separate features (X) and labels (y)
    X = data.iloc[:, 1:].values  # Features (exclude the first column, which is the label)
    y = data.iloc[:, 0].values   # Labels (first column)

    # Split data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForestClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Print model accuracy
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

    # Save the model to the specified file
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}.")

# Run the training function
train_model(data_file="backend/datasets/combined_dataset.csv", model_file="backend/models/hand_gesture_model.pkl")
