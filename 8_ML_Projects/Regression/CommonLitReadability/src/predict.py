import utils

import joblib
import pandas as pd
import numpy as np

# Load the saved model
best_model = joblib.load("models/linear_regression_0.bin")
# Load the saved vectorizer
vectorizer = joblib.load("models/vectorizer_linear_regression_0.bin")

# Load the test data
df = pd.read_csv("input/test.csv")
# Clean the test data
X_test = df["excerpt"].apply(utils.clean_text)
# Transform the test data
X_test = vectorizer.transform(X_test)

o_pred = best_model.predict(X_test)
output_df = pd.DataFrame({'id': df['id'], 'target': o_pred})
output_df.to_csv('output.csv', index=False)
