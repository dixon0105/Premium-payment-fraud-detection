from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

def feature_selection(X, y):
    # Feature importance-RandomForest
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    rf_importances = rf_model.feature_importances_

    # Feature importance-XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)
    xgb_importances = xgb_model.get_booster().get_score(importance_type='weight')
    xgb_importances = pd.Series(xgb_importances).reindex(X.columns, fill_value=0)

    # Lasso for feature selection
    lasso = LassoCV()
    lasso.fit(X, y)
    lasso_importances = np.abs(lasso.coef_)

    # Mutual Information
    mi = mutual_info_classif(X, y)

    # Combine all importances
    combined_importances = (rf_importances + xgb_importances + lasso_importances + mi) / 4

    # Select top features based on combined importances
    top_features = X.columns[np.argsort(combined_importances)[-4:]]
    return top_features
