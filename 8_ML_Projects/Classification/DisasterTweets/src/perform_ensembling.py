import config
import utils

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Define the path to the folder containing the models and vectorizers
model_dir = Path(config.MODEL_OUTPUT)

# Initialize lists to store the models and vectorizers
models = []
vectorizers = []

# Iterate through all the files in the model folder
for file in os.listdir(model_dir):
    # Check if the file is a model with the .bin extension
    if file.endswith(".bin") and "vectorizer" not in file:
        # Load the model and add it to the list
        models.append(joblib.load(os.path.join(model_dir, file)))
    # Check if the file is a vectorizer with the .bin extension
    elif file.endswith(".bin") and "vectorizer" in file:
        # Load the vectorizer and add it to the list
        vectorizers.append(joblib.load(os.path.join(model_dir, file)))

# Load the test data
df = pd.read_csv("input/test.csv")
# Clean the test data
X_test = df["text"].apply(utils.clean_text)

# Initialize a list to store the predictions
predictions = []

# Iterate through the models and vectorizers
for model, vectorizer in zip(models, vectorizers):
    print(model)
    # Transform the test data using the corresponding vectorizer
    X_test_transformed = vectorizer.transform(X_test)
    # Perform predictions using the corresponding model
    pred = model.predict(X_test_transformed)
    # Append the predictions to the list
    predictions.append(pred)

# Average the predictions from all the models to get the final prediction
final_pred = np.mean(predictions, axis=0)
final_pred = np.round(final_pred)
output_df = pd.DataFrame({'id': df['id'], 'target': final_pred})
output_df.to_csv('ensemble_output.csv', index=False)