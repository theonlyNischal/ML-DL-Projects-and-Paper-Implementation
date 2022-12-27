from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
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
    if model_name == "logistic_regression":
        # Create a LogisticRegression estimator
        model = LogisticRegression(random_state=0, tol=1e-5)
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        grid_param = dict(solver=solvers,penalty=penalty,C=c_values)
        
    elif model_name == "random_forest":
        # Create a RandomForestClassifier estimator
        model = RandomForestClassifier(random_state=0, n_jobs=-1)

        # Update the grid of hyperparameters
        grid_param = {
            # 'tfidf__min_df': [5, 10],
            # 'tfidf__ngram_range': [(1, 3), (1, 6)],
            'n_estimators': [10, 50],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
    elif model_name == "linear_svc":
        model = LinearSVC(random_state=0, tol=1e-5)
        grid_param = {
            # 'tfidf__min_df': [5, 10],
            # 'tfidf__ngram_range': [(1, 3), (1, 6)],
            'C': [1, 100],
            'loss': ['hinge']
        }
    
    elif model_name == "multinomial_nb":
        model = MultinomialNB()
        grid_param = {
            'alpha': [0.01, 0.1, 1, 10],
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