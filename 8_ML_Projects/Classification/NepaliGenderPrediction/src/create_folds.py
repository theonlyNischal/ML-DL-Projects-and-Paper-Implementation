import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # Load the data from input/train.csv
    df = pd.read_csv("input/all_name.csv")
    len_before = len(df)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=["Name", "Gender"])
    df = df.reset_index(drop=True)
    print(f"Removed {(len_before-len(df))/len_before:.2%} duplicates.")
    # Add a new column called "kfold" with a default value of -1.
    df["kfold"] = -1
    # Fetch the target
    y = df["kfold"].values
    # Initiate a kfold class
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Iterate over the folds and split the data into train and validation sets
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, "kfold"] = fold
        
    # Save the modified dataframe into a csv file
    df.to_csv("input/train_folds.csv", index=False)


    
