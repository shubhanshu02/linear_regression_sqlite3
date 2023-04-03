import sqlite3
import vowpalwabbit


class LinearRegressionModel:
    """
    Linear regression model that uses SQLite3 database as a datasource
    """

    def __init__(self, model=None, db_path=None):
        self.model = model
        self.conn = sqlite3.connect(db_path)

        if self.model is None:
            self.model = vowpalwabbit.Workspace(quiet=True)

        self.conn.create_function(
            "train_example", 8, self._train_example, deterministic=True
        )
        self.conn.create_function(
            "predict_example", 8, self._predict_example, deterministic=True
        )

    def __del__(self):
        """
        Destructor to remove the UDF and close the database connection
        """
        self.conn.create_function("db_to_vw", 8, None)
        self.conn.close()

    def train(self, data_query):
        """
        Training Function to learn from the given examples in the database
        """
        example_cursor = self.conn.execute(data_query)
        example_cursor.fetchall()

    def _train_example(
        self, price, sqft, age, feature4, label, weight, tag, initial_prediction
    ):
        '''
        UDF for in-database model training
        '''
        example = self._db_to_vw(
            price, sqft, age, feature4, label, weight, tag, initial_prediction
        )
        self.model.learn(example)

    def _predict_example(
        self, price, sqft, age, feature4, label, weight, tag, initial_prediction
    ):
        '''
        UDF for in-database prediction
        '''
        example = self._db_to_vw(
            price, sqft, age, feature4, label, weight, tag, initial_prediction
        )

        pred = "{:.4f}".format(self.model.predict(example))
        if tag is not None:
            pred += f" {tag}"
        return pred

    def _db_to_vw(
        self, price, sqft, age, feature4, label, weight, tag, initial_prediction
    ):
        """
        Function to convert the database tuples into a VW example
        """
        example = ""

        if label is not None:
            example += f"{label} "

        if weight is not None:
            example += f"{weight} "

        if initial_prediction is not None:
            example += f"{initial_prediction} "

        if tag is not None:
            example += f"'{tag} "

        example += f"| price:{price} sqft:{sqft} age:{age} {feature4}"

        return example

    def predict(self, data_query):
        """
        Predicts the label for the given example in the database
        """
        prediction_cursor = self.conn.execute(data_query)
        preds = [pred[0] for pred in prediction_cursor]
        return preds
