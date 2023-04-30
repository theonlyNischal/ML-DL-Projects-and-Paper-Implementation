import utils

import joblib
import pandas as pd
import numpy as np

# Load the saved model
best_model = joblib.load("models/random_forest_0.bin")
# Load the saved vectorizer
# vectorizer = joblib.load("models/vectorizer_linear_regression_0.bin")

# Load the test data
df = pd.read_csv("input/test.csv")
df_columns_to_predict = ["id","minimum_nights", "number_of_reviews", "noise(dB)"]
df = df[df_columns_to_predict]
# Clean the test data
# X_test = df["excerpt"].apply(utils.clean_text)
# Transform the test data
X_test = df

o_pred = best_model.predict(X_test)
submission_format = pd.read_csv("input/submission_format.csv", index_col='Id')

my_submission = pd.DataFrame(data=o_pred,
                             columns=submission_format.columns,
                             index=submission_format.index)

print(my_submission.head())
my_submission.to_csv('submission.csv')
