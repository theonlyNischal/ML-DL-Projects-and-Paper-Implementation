from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV


def perform_hyperparameter_search(X_train, y_train, model_name):
    """
    Performs hyperparameter search using GridSearchCV.
    
    Parameters:
        - X_train: Numpy array of feature vectors
        - y_train: Numpy array of labels
        - model: str, name of the model to use (logistic_regression, random_forest, svc)
    
    Returns:
        - best_estimator: fitted model with the best hyperparameters
    """
    # Define the model and hyperparameters to search
    if model_name == "linear_regression":
        # Create a LogisticRegression estimator
        model = LinearRegression()
        grid_param = {}
    elif model_name == "random_forest":
        # Create a RandomForestClassifier estimator
        model = RandomForestRegressor(random_state=0, n_jobs=-1)

        # Update the grid of hyperparameters
        grid_param = {
            # 'tfidf__min_df': [5, 10],
            # 'tfidf__ngram_range': [(1, 3), (1, 6)],
            'n_estimators': [10, 50],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }

    # Create an instance of GridSearchCV
    gridSearchProcessor = GridSearchCV(
        estimator=model,
        param_grid=grid_param,
        cv=5,
        scoring="accuracy",
        refit=True
    )
    # Fit the grid search to the training data
    gridSearchProcessor.fit(X_train, y_train)
    # Return the best estimator
    best_model = gridSearchProcessor.best_estimator_
    return best_model