from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression(random_state=42):
    return LogisticRegression(max_iter=1000, random_state=random_state)

def get_random_forest(random_state=42):
    return RandomForestClassifier(n_estimators=200, random_state=random_state)