from sklearn.model_selection import train_test_split

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def split_data(X, y):

    return train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


def predict_customer(model, customer_data):

    return model.predict(customer_data)


def main():

    X, y = prepare_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    loaded_model = load_model()

    prediction = predict_customer(loaded_model, X_test.iloc[[0]])

    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
