import json
import os

from .train import Model


def main():
    """
    Main function that loads data from environment variables, uses a pre-trained model
    to make predictions, and stores the results in a JSON file.

    This function performs the following steps:
    
    1. Instantiates a `Model` object from the `train` module.
    2. Retrieves the data from the "DATA" environment variable.
    3. Parses the data as JSON.
    4. Uses the model to make predictions on the provided data.
    5. Creates a list of records containing dataset information, model architecture, 
       evaluation score, input data, and predicted labels.
    6. Stores the results in a file named `out.json`.

    Raises:
        ValueError: If the "DATA" environment variable is not set or no data is provided.

    Example:
        Assuming you have a valid JSON in the "DATA" environment variable:
        
        ```bash
        export DATA='[[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4]]'
        python script.py
        ```

        The output will be saved in `out.json`. You can use `os.environ['DATA']` to set the environment variable.

    Output JSON structure:
        The output will be a JSON array of objects, where each object contains:

        - dataset (str): The name of the dataset used by the model (e.g., "iris").
        - architecture (str): The name of the model's architecture (e.g., "KNN").
        - features (float): The model's evaluation score on the test dataset.
        - data (list): The input data (features) for the prediction.
        - label (str): The predicted label for the corresponding data.

    Example output:
    
    .. code-block:: json

        [
            {
                "dataset": "iris",
                "architecture": "KNN",
                "features": 0.96,
                "data": [5.1, 3.5, 1.4, 0.2],
                "label": "setosa"
            }
        ]

    Returns:
        None
    """
    m = Model()
    data = os.getenv("DATA")
    if not data:
        raise ValueError("No data provided")

    data = json.loads(data)
    records = [
        {
            "dataset": m.dataset,
            "architecture": m.architecture,
            "features": m.eval,
            "data": record,
            "label": label,
        }
    for record, label in zip(data, m(data))]

    json.dump(records, open("out.json", "w"))