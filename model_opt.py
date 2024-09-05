import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from features_selection import feature_selection

def train_and_optimize_models(data):
    # Preprocess your data
    X = data.drop('fraud_label', axis=1)  # Features
    y = data['fraud_label']  # Target

    # Select features
    selected_features = feature_selection(X, y)
    X_selected = X[selected_features]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Optuna
    def rf_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        return cross_val_score(rf_model, X_train, y_train, cv=3, scoring='accuracy').mean()

    def gb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)

        gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        return cross_val_score(gb_model, X_train, y_train, cv=3, scoring='accuracy').mean()

    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

        xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        return cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='accuracy').mean()

    # Optimize-Random Forest
    rf_study = optuna.create_study(direction='maximize')
    rf_study.optimize(rf_objective, n_trials=50)
    rf_best_params = rf_study.best_params

    # Optimize-Gradient Boosting
    gb_study = optuna.create_study(direction='maximize')
    gb_study.optimize(gb_objective, n_trials=50)
    gb_best_params = gb_study.best_params

    # Optimize-XGBoost
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=50)
    xgb_best_params = xgb_study.best_params

    # Train models
    rf_model = RandomForestClassifier(**rf_best_params, random_state=42)
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier(**gb_best_params, random_state=42)
    gb_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(**xgb_best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Evaluate models
    rf_accuracy = cross_val_score(rf_model, X_test, y_test, cv=3, scoring='accuracy').mean()
    gb_accuracy = cross_val_score(gb_model, X_test, y_test, cv=3, scoring='accuracy').mean()
    xgb_accuracy = cross_val_score(xgb_model, X_test, y_test, cv=3, scoring='accuracy').mean()

    # Plot performance
    performance_data = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost'],
        'Accuracy': [rf_accuracy, gb_accuracy, xgb_accuracy]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=performance_data)
    plt.title('Model Performance Comparison')
    plt.savefig('model_performance.png')

    return rf_model, gb_model, xgb_model, selected_features
