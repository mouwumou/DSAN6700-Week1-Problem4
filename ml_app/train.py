from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Model:
    """
    A simple machine learning model class using K-Nearest Neighbors (KNN) on the Iris dataset.

    This class trains a KNN classifier on the Iris dataset and provides functionality 
    to make predictions on new data. The class also evaluates the model's performance 
    using accuracy score on a test dataset split.

    Attributes:
        dataset (str): The name of the dataset used, set to "iris".
        architecture (str): The architecture/model type, set to "KNN".
        features (list): List of feature names in the dataset.
        labels (list): List of target label names in the dataset.
        model (KNeighborsClassifier): The trained KNN classifier model.
        eval (float): The evaluation accuracy score of the model on the test dataset.

    Args:
        test_size (float, optional): The proportion of the dataset to include in the test split. 
            Defaults to 0.5 (50% training, 50% testing).
    """

    def __init__(self, test_size=0.5):
        """
        Initializes the Model class by setting up the dataset, training the model, and
        evaluating its accuracy.

        Args:
            test_size (float, optional): The proportion of the dataset to use for testing. 
                Defaults to 0.5.
        """
        self.dataset = "iris"
        self.architecture = "KNN"
        self._train(test_size)

    def __call__(self, data):
        """
        Make predictions on new data.

        Args:
            data (iterable): A collection of records to be classified. Each record should be 
                a list or array of numerical values representing the features.

        Yields:
            str: Predicted label for each input record.

        Raises:
            ValueError: If the input data record is not correctly formatted or does not match 
                the expected feature length.
        """
        for record in data:
            if len(record) != len(self.labels) and not all([isinstance(val, float) or isinstance(val, int) for val in record]):
                raise ValueError(f"Malformed data record {record}")

        yield from (self.labels[label] for label in self.model.predict(data))

    def _init_data(self, test_size=0.5):
        """
        Initialize and split the Iris dataset into training and testing sets.

        Args:
            test_size (float, optional): The proportion of the dataset to use for testing. 
                Defaults to 0.5.
        """
        iris_dataset = load_iris()
        self.features = iris_dataset.feature_names
        self.labels = iris_dataset.target_names
        x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.5)
        self._train_data = (x_train, y_train)
        self._eval_data = (x_test, y_test)

    def _score(self):
        """
        Evaluate the model's accuracy on the test dataset.

        Returns:
            float: The accuracy score of the model.
        """
        preds = self.model.predict(self._eval_data[0])
        return accuracy_score(preds, self._eval_data[1])

    def _train(self, test_size=0.5):
        """
        Train the KNN classifier on the training data.

        Args:
            test_size (float, optional): The proportion of the dataset to use for testing. 
                Defaults to 0.5.
        """
        self._init_data()
        classifier = KNeighborsClassifier()
        classifier.fit(*self._train_data)
        self.model = classifier
        self.eval = self._score()