import pandas as pd

train_values = pd.read_csv("input/train_values.csv")
train_labels = pd.read_csv("input/train_labels.csv")

train_df = pd.merge(train_labels, train_values, on="building_id")

train_df.to_csv("input/train.csv", index=False)