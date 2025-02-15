#inferencing the machine learning model
import pickle
import numpy as np
from sklearn.datasets import load_iris


# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Load the trained model and make inference
def predict(input_features):
    with open("iris_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    prediction = loaded_model.predict([input_features])
    return iris.target_names[prediction[0]]

# Example input for inference
sample_input = np.array([5.1, 0.5, 1.4, 0.2])
result = predict(sample_input)
print(f"Predicted class: {result}")
