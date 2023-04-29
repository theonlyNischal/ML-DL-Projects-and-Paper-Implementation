import utils

import joblib
import pandas as pd
import numpy as np

# Load the saved model
best_model = joblib.load("models/random_forest_0.bin")
# Load the saved vectorizer
# vectorizer = joblib.load("models/vectorizer_linear_regression_0.bin")

columns_to_predict = ["building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id", "count_floors_pre_eq", "age", "area_percentage", "height_percentage", "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone", "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone", "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick", "has_superstructure_timber", "has_superstructure_bamboo", "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered", "has_secondary_use_agriculture", "has_secondary_use_hotel", "has_secondary_use_rental", "has_secondary_use_institution"]

# Load the test data
df = pd.read_csv("input/test_values.csv")
df = df[columns_to_predict]
# Clean the test data
# X_test = df["excerpt"].apply(utils.clean_text)
# Transform the test data
# X_test = vectorizer.transform(X_test)
X_test = df

o_pred = best_model.predict(X_test)
submission_format = pd.read_csv("input/submission_format.csv", index_col='building_id')

my_submission = pd.DataFrame(data=o_pred,
                             columns=submission_format.columns,
                             index=submission_format.index)

print(my_submission.head())
my_submission.to_csv('submission.csv')
