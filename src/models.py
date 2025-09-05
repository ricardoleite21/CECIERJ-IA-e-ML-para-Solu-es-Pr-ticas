"""
Modelos de ML (classificação e regressão) com Scikit-learn.
"""
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

def make_classifiers(random_state: int = 42) -> Dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=200, random_state=random_state),
        "knn": KNeighborsClassifier(),
        "tree": DecisionTreeClassifier(random_state=random_state),
    }

def make_regressors(random_state: int = 42) -> Dict[str, Any]:
    return {
        "linreg": LinearRegression(),
        "rf": RandomForestRegressor(random_state=random_state),
    }
