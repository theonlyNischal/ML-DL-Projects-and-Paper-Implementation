from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier

models = {
    "logistic_regression": LogisticRegression(),
    "linear_svc": LinearSVC(),
    "random_forest": RandomForestClassifier(),
    # "xgb": XGBClassifier(
    #     use_label_encoder=False,
    #     eval_metric="logloss"
    # ),
    "decision_tree_gini": DecisionTreeClassifier(
        criterion = "gini"
    ),
    "decision_tree_entropy": DecisionTreeClassifier(
        criterion = "entropy"
    )
}