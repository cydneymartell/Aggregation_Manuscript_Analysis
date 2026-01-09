import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso, Ridge
import argparse 
import pandas as pd
import os
import json 

def get_train_test_sets(df,selected_features, tgt,scale=True):
    X_train = df.query(""" split == "train" or split == "valid" """)[selected_features]
    y_train = df.query(""" split == "train" or split == "valid" """)[tgt]
    X_test = df.query(""" split == "test"  """)[selected_features]
    y_test = df.query(""" split == "test"  """)[tgt]
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

def bootstrap_ci(model, X_test, y_test, metric_func, n_bootstrap=1000, ci=95, random_state=None):
    rng = np.random.RandomState(random_state)
    n = len(y_test)
    scores = []
    y_pred = model.predict(X_test)
    
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        # For R², use y_pred and y_test; for Spearman, use y_pred and y_test as well
        score = metric_func(y_test.iloc[idx], y_pred[idx])

        scores.append(score)
    
    scores = np.array(scores)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper, np.mean(scores)

def spearmanr_only(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def train_test_model_ridge(selected_features, df, tgt,a,bootstrap=True):
    #get splits and scale
    X_train, y_train, X_test, y_test=get_train_test_sets(df,selected_features,tgt)
    
    #ridge model
    model = Ridge(alpha=a)
    final_model = model.fit(X_train, y_train)

    # Test the model on the test set
    y_pred_test = final_model.predict(X_test)
    plt.figure(figsize=(1,1))
    sns.scatterplot(y=y_pred_test, x=y_test, alpha = 0.2, color = "slateblue")
    plt.title("Test")
    plt.ylabel("predicted log2 fold change")
    r2 = r2_score(y_test,y_pred_test)
    r2_text = f"$R^2 = {r2:.2f}$"
    plt.legend([r2_text], loc="upper left")
    plt.show()
    
#     print("test r2",r2_score(y_test,y_pred_test))
#     print('mse',mean_squared_error(y_test,y_pred_test))
#     print('spearman',spearmanr(y_test,y_pred_test) )
    
    if bootstrap:
        r2_lower, r2_upper, r2_mean = bootstrap_ci(final_model, X_test, y_test, r2_score, n_bootstrap=1000, ci=95, random_state=42)
        print(f"R²: mean={r2_mean:.4f}, 95% CI=({r2_lower:.4f}, {r2_upper:.4f})")

        spearman_lower, spearman_upper, spearman_mean = bootstrap_ci(final_model, X_test, y_test, spearmanr_only, n_bootstrap=1000, ci=95, random_state=42)
        print(f"Spearman’s ρ: mean={spearman_mean:.4f}, 95% CI=({spearman_lower:.4f}, {spearman_upper:.4f})")

    return [r2_lower,r2_upper,r2_mean], [spearman_lower, spearman_upper, spearman_mean]

def main():
    parser = argparse.ArgumentParser(description="Train Ridge Regression")
    parser.add_argument("--data_csv", required=True, help="CSV file with training validation and test data")
    parser.add_argument("--feature_file", required=True, help="File containing feature names, one per line")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument("--output_dir", help="directory to save model")
    parser.add_argument("--alpha", type=int, default=20, help="Number of iterations for RandomizedSearchCV")
    
    args = parser.parse_args()

    # Load data
    meta_data = pd.read_csv(args.data_csv)
    with open(args.feature_file, "r") as f:
        features = [line.strip() for line in f.readlines()]
    tgt = args.target
    alpha = args.alpha
    
    r2_ci, spearmanr_ci =  train_test_model_ridge(features,meta_data,tgt,alpha,True)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.join(out_dir, f"{tgt}_ridge_model_results.json")

    # Convert everything to lists (in case they are numpy arrays)
    data_to_save = {
        "r2_ci": r2_ci.tolist() if hasattr(r2_ci, "tolist") else r2_ci,
        "spearmanr_ci": spearmanr_ci.tolist() if hasattr(spearmanr_ci, "tolist") else spearmanr_ci
    }

    with open(filename, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"Saved R2 and Spearman results to {filename}")

if __name__ == "__main__":
    main()