
"""
Train and evaluate an XGBoost model with RandomizedSearchCV to tune hyperparameters

Example usage:
    python train_xgboost.py 
        --data_csv data.csv 
        --feature_file features.txt 
        --target log2_fold_change_75_clip 
        --output_model best_xgb_model.json
"""

import argparse
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform, spearmanr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost with RandomizedSearchCV")
    parser.add_argument("--data_csv", required=True, help="CSV file with training validation and test data")
    parser.add_argument("--feature_file", required=True, help="File containing feature names, one per line")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument("--output_model", default="best_xgb_model.json", help="Filename to save best model")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()

    # Load data
    meta_data = pd.read_csv(args.data_csv)
    meta_valid_train = meta_data.query(""" split == "train" or split == "valid" """)
    meta_test = meta_data.query(""" split == "test" """)
    
    # Load feature list
    with open(args.feature_file, "r") as f:
        features = [line.strip() for line in f.readlines()]
    
    # Split features and target
    X_train = meta_valid_train[features]
    y_train = meta_valid_train[args.target]
    X_test = meta_test[features]
    y_test = meta_test[args.target]


    # Define parameter distributions for RandomizedSearch
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.005, 0.1),
        'n_estimators': randint(200, 800),
        'subsample': uniform(0.5, 0.2),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.2),
        'min_child_weight': randint(1, 10),
        'alpha': uniform(0, 10000)
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=args.random_state)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=args.cv,
        scoring='neg_mean_squared_error',
        random_state=args.random_state,
        verbose=1
    )

    # Fit model
    print("Training XGBoost model with RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Parameters (RandomizedSearchCV): {}".format(best_params))

    best_model = random_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    # Evaluation
    print("\nEvaluation Metrics:")
    print('Train R2: {:.4f}'.format(r2_score(y_train, y_pred_train)))
    print('Train MSE: {:.4f}'.format(mean_squared_error(y_train, y_pred_train)))
    print('Test R2: {:.4f}'.format(r2_score(y_test, y_pred_test)))
    print('Test MSE: {:.4f}'.format(mean_squared_error(y_test, y_pred_test)))
    print('Test Spearman: {:.4f}'.format(spearmanr(y_test, y_pred_test)[0]))

    # Scatter plot
    sns.scatterplot(x=y_test, y=y_pred_test)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("XGBoost Predictions vs Observed")
    plt.tight_layout()
    plt.show()

    # Save best model
    best_model.get_booster().save_model(args.output_model)
    print("\nBest model saved to {}".format(args.output_model))



if __name__ == "__main__":
    main()
