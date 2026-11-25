import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task
from sklearn.model_selection import train_test_split

from main import (
    evaluate_model,
    load_model,
    predict_customer,
    prepare_data,
    save_model,
    train_model,
)

# Files used by code quality and security tasks
PROJECT_FILES = [
    "model_pipeline.py",
    "main.py",
    "pipeline_prefect.py",
]


# ==========================================
# TASK 1: INSTALL DEPENDENCIES
# ==========================================


@task(name="Install dependencies")
def install_dependencies():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )  # nosec B603
    print("Dependencies installed successfully!")


# ==========================================
# TASK 2: FORMAT CODE
# ==========================================


@task(name="Format code")
def format_code():
    subprocess.run(
        [sys.executable, "-m", "black", "--line-length", "79"] + PROJECT_FILES,
        check=True,
    )  # nosec B603
    print("Code formatting completed!")


# ==========================================
# TASK 3: CHECK CODE QUALITY
# ==========================================


@task(name="Check code quality")
def check_code_quality():
    subprocess.run(
        [sys.executable, "-m", "flake8"] + PROJECT_FILES,
        check=True,
    )  # nosec B603
    print("Code quality check completed!")


# ==========================================
# TASK 4: CHECK CODE SECURITY
# ==========================================


@task(name="Check code security")
def check_security():
    subprocess.run(
        [sys.executable, "-m", "bandit", "-q", "-r"] + PROJECT_FILES,
        check=True,
    )  # nosec B603
    print("Security check completed!")


# ==========================================
# TASK 5: RUN UNIT TESTS
# ==========================================


@task(name="Run unit tests")
def run_unit_tests():
    subprocess.run(
        [sys.executable, "-m", "pytest", "test_model_pipeline.py", "-v"],
        check=True,
    )  # nosec B603
    print("Unit tests completed!")


# ==========================================
# TASK 6: PREPARE DATA
# ==========================================


@task(name="Prepare data")
def prepare_data_task():
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    print("Data preparation completed!")
    return X_train, X_test, y_train, y_test


# ==========================================
# TASK 7: TRAIN MODEL
# ==========================================


@task(name="Train model")
def train_model_task(X_train, y_train):
    model = train_model(X_train, y_train)
    print("Model training completed!")
    return model


# ==========================================
# TASK 8: SAVE MODEL
# ==========================================


@task(name="Save model")
def save_model_task(model):
    save_model(model)
    print("Model saved!")


# ==========================================
# TASK 9: LOAD MODEL
# ==========================================


@task(name="Load model")
def load_model_task():
    model = load_model()
    print("Model loaded!")
    return model


# ==========================================
# TASK 10: EVALUATE MODEL
# ==========================================


@task(name="Evaluate model")
def evaluate_model_task(model, X_test, y_test):
    accuracy = evaluate_model(model, X_test, y_test)
    print("Model evaluation completed!")
    return accuracy


# ==========================================
# TASK 11: MAKE PREDICTION
# ==========================================


@task(name="Make prediction")
def prediction_task(model, X_test):
    customer = X_test.iloc[[0]]
    prediction = predict_customer(model, customer)
    print("Customer prediction:", prediction)
    return prediction


# ==========================================
# FLOW 1: CODE
# ==========================================


@flow(name="code")
def code():
    install_dependencies()
    format_code()
    check_code_quality()
    check_security()
    run_unit_tests()
    print("All code checks completed!")


# ==========================================
# FLOW 2: TRAIN
# ==========================================


@flow(name="train")
def train():
    X_train, _, y_train, _ = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    print("Training flow completed!")


# ==========================================
# FLOW 3: EVALUATE
# ==========================================


@flow(name="evaluate")
def evaluate():
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = load_model_task()
    accuracy = evaluate_model_task(model, X_test, y_test)
    print("Evaluation accuracy:", accuracy)


# ==========================================
# FLOW 4: ALL
# ==========================================


@flow(name="all")
def all():
    # First execute code checks
    code()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_task()

    # Train model
    model = train_model_task(X_train, y_train)

    # Save model
    save_model_task(model)

    # Load model
    loaded_model = load_model_task()

    # Evaluate model
    evaluate_model_task(loaded_model, X_test, y_test)

    # Make prediction
    prediction_task(loaded_model, X_test)

    print("Complete ML pipeline finished!")


# ==========================================
# MAIN: SELECT FLOW FROM TERMINAL
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Churn Modelling Prefect Pipeline"
    )
    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate", "code"],
        default="all",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all()
    elif args.flow in ["train", "entrainement"]:
        train()
    elif args.flow == "evaluate":
        evaluate()
    elif args.flow == "code":
        code()
