import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def prepare_data(data_filename="Churn_Modelling.csv"):

    df = pd.read_csv(data_filename)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    df = df.drop(["Surname", "Geography"], axis=1)

    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    X = X.drop(["RowNumber", "CustomerId"], axis=1)

    return X, y


def train_model(X_train, y_train):

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)

    return rf


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy * 100, "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


def save_model(model, filename="classifier.joblib"):

    joblib.dump(model, filename)

    print("Model saved successfully!")


def load_model(filename="classifier.joblib"):

    model = joblib.load(filename)

    print("Model loaded successfully!")

    return model
