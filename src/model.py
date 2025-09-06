import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import os

X_train = pd.read_csv(r"../data/processed/X_train.csv")
X_test = pd.read_csv(r"../data/processed/X_test.csv")
y_train = pd.read_csv(r"../data/processed/y_train.csv")
y_test = pd.read_csv(r"../data/processed/y_test.csv")

def train(model_save_path="model.pkl"):
    print('Loading data...')

    param_grid = {
        'n_estimators': [200, 600],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=1),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train.values.ravel())
    model = grid_search.best_estimator_

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.3f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_save_path}")

    return model, accuracy


if __name__ == "__main__":
    model_path = "../data/processed/rf_model.pkl"
    model, accuracy = train(model_path)
    print("Training complete, model saved to disk.")
