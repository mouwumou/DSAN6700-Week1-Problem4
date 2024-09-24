Project Name
============

A machine learning project using the K-Nearest Neighbors (KNN) algorithm
to classify data from the Iris dataset. This project includes
functionality to make predictions on new data, evaluate the model, and
save the predictions in a JSON file.

Table of Contents
-----------------

-  `Installation <#installation>`__
-  `Tutorial <#tutorial>`__
-  `Reference <#reference>`__
-  `Explanation <#explanation>`__

Installation
------------

Requirements
~~~~~~~~~~~~

Ensure you have Python 3.10+ installed. You’ll also need the following
Python libraries:

-  ``scikit-learn``

To install the required libraries, you can use ``pip``:

.. code:: bash

   pip install scikit-learn

Or manage your dependencies using poetry:

.. code:: bash

   poetry install

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

Clone the project repository to your local machine:

.. code:: bash

   git clone https://github.com/your-repo/project-name.git

Navigate to the project directory:

.. code:: bash

   cd project-name

Tutorial
--------

This section provides a step-by-step guide on how to run the project and
use the provided scripts.

Step 1: Prepare the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that your environment is ready by setting the necessary ``DATA``
environment variable. This variable should contain the JSON data for the
model to process.

For example, you can export a simple test dataset in JSON format like
this:

.. code:: bash

   `export DATA='[[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4]]'`

Or setup by running the following command:

.. code:: python

   import os
   os.environ['DATA'] = '[[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4]]'

Step 2: Run the Model
~~~~~~~~~~~~~~~~~~~~~

Once the ``DATA`` environment variable is set, run the ``main()``
function in the script:

.. code:: bash

   python -m ml_app

The output will be saved in a file named ``out.json``, which will
contain the predicted labels for the provided input data.

Step 3: View the Output
~~~~~~~~~~~~~~~~~~~~~~~

Check the ``out.json`` file generated in the root of the project to see
the results of the model’s predictions:

``[{"dataset": "iris", "architecture": "KNN", "features": 0.96, "data": [5.1, 3.5, 1.4, 0.2], "label": "setosa"}]``

How to Modify the Test Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the model uses 50% of the Iris dataset for training and 50%
for testing. If you want to change this split, you can pass a different
``test_size`` argument to the ``Model`` class in ``train.py``.

Reference
---------

This section provides a detailed reference for the key components of the
project.

``Model`` Class (in ``train.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **``__init__(test_size=0.5)``**: Initializes the model with a
   specified train/test split. Trains the KNN classifier on the Iris
   dataset.
-  **``__call__(data)``**: Makes predictions on the provided data.
-  **``_init_data(test_size)``**: Splits the dataset into training and
   testing sets.
-  **``_train(test_size)``**: Trains the KNN classifier.
-  **``_score()``**: Returns the accuracy score of the model on the test
   data.

``main()`` Function (in ``script.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Loads the data from the ``DATA`` environment variable, uses the
   trained model to make predictions, and saves the results to
   ``out.json``.
-  Raises a ``ValueError`` if no data is provided.

Explanation
-----------

This section provides a deeper understanding of the project, its
structure, and its purpose.

Why Use K-Nearest Neighbors (KNN)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNN is a simple and effective classification algorithm. It works by
finding the ``k`` closest data points (neighbors) to the input and
assigning the label that occurs most frequently among them. This project
uses KNN for its simplicity and interpretability, making it an ideal
choice for small datasets like the Iris dataset.

Structure of the Iris Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Iris dataset is a well-known dataset in machine learning. It
contains 150 samples from three species of Iris flowers: *setosa*,
*versicolor*, and *virginica*. Each sample has four features:

[Sepal length, Sepal width, Petal length, Petal width]

These features are used to classify the species of the iris flower.

Project Architecture
~~~~~~~~~~~~~~~~~~~~

-  ``train.py``: Contains the ``Model`` class which is responsible for
   training the KNN model, making predictions, and evaluating the model.
-  ``script.py``: The main entry point of the project. It loads data,
   uses the ``Model`` class to make predictions, and saves the results
   to ``out.json``.

This project demonstrates the entire machine learning workflow, from
training to inference, with the Iris dataset using the KNN algorithm.
