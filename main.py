from linear_regression import LinearRegressionModel


if __name__ == "__main__":
    # Learn from the database examples
    model = LinearRegressionModel(db_path="house_dataset.db")
    model.train(
        data_query="select train_example(price,sqft,age,feature4,label,weight,tag,initial_prediction) "
        "FROM house_dataset where label is not NULL"
    )

    # Predict the label for the given example
    predictions = model.predict(
        data_query="select predict_example(price,sqft,age,feature4,label,weight,tag,initial_prediction) "
        "FROM house_dataset where label is NULL"
    )
    print(predictions)
