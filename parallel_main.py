import json
import os
import time
import glob
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from classification.src import classifiers as cl
from featureselection.src.feature_selection_methods import *
from Pipeline import load_graph, load_full_corr, train_and_evaluate, cross_validate_model, print_selected_features, failsafe_feature_selection, classify, load_dataframe
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from featureselection.src import cluster

# ========== CONFIGURATION ========== #

classifiers_to_run = ["SVM", "RandomForest", "LogR", "LDA", "KNN"]

feature_selection_methods = [
    ("Lasso_selection", Lasso_selection, {"alpha": 0.044984, "max_iter": 2000}, "cv"), #0.044984 for full corr
    ("HSIC_Lasso", hsiclasso, {"num_feat": 19}, "cv"), #98 for full corr
    ("mRMR", mRMR, {"num_features_to_select": 100}, "cv"),
    ("Permutation", Perm_importance, {}, "cv"),
    ("forwards SFS", forwards_SFS, {"n_features_to_select": 20}, "train"),
    ("backward SFS", backwards_SFS, {"n_features_to_select": 10}, "train")
]

inf_methods = ["partial_corr", "mutual_info", "norm_laplacian", "rlogspect"]
#("ReliefF", reliefF_, {"num_features_to_select": 200}, "cv")
# ========== SAVE RESULTS ========== #

def save_results(classifier, feature_selection_name, inf_method, results_dict):
    os.makedirs(f"results_graph_NYU_male/{classifier}/{inf_method}", exist_ok=True)
    
    # Generate timestamp: YYYYMMDD-HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filename = f"results_graph_NYU_male/{classifier}/{inf_method}/{feature_selection_name}_{timestamp}.json"
    
        # Helper function to convert numpy types
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=4, default=convert)
    
    print(f"✅ Saved: {filename}")

# ========== MAIN PER CLASSIFIER ========== #

def main_for_classifier(classifier):
    print(f"\n\n========== Running pipeline for {classifier} ==========\n")
    
    for inf_method in inf_methods:
        print(f"\n--- Using inference method: {inf_method} ---")
        
        # Load data with current inference method
        X, y = load_graph(sex='male', site_id='NYU', method=inf_method)

        # Run raw model just for reference
        X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

        # Use clustered features for Perm / SFS
        X_clustered = cluster.cluster(X_train, y_train, t=3)
        X_mRMR = mRMR(X_train, y_train, classifier, num_features_to_select=50)

        # Add results for raw data without feature selection
        print(f"\n=== Running Raw data for {classifier} ===")
        result_raw = {
            "classifier": classifier,
            "feature_selection": "Raw data",
            "mode": "cv"
        }

        start_time = time.time()

        # Cross-validation for raw data
        selected_features, selected_feature_names, avg_metrics, fold_metrics = cross_validate_model(
            X, y, None, classifier, n_splits=5, return_metrics=True
        )
        result_raw["selected_features"] = selected_features
        result_raw["selected_feature_names"] = list(selected_feature_names) if selected_feature_names is not None else []
        result_raw["metrics"] = avg_metrics
        result_raw["fold_metrics"] = fold_metrics

        # Save raw data results
        elapsed = time.time() - start_time
        result_raw["elapsed_seconds"] = elapsed
        save_results(classifier, "Raw data", inf_method, result_raw)

        # Loop over feature selection methods with tqdm per classifier
        for fs_name, fs_func, fs_kwargs, mode in tqdm(feature_selection_methods, desc=f"{classifier} pipeline", position=0, leave=True):
            print(f"\n=== Running {fs_name} for {classifier} ===")
            
            result = {
                "classifier": classifier,
                "feature_selection": fs_name,
                "mode": mode
            }

            start_time = time.time()

            if mode == "cv":
                # Normal cross-validation → capture avg metrics
                selected_features, selected_feature_names, avg_metrics, fold_metrics = cross_validate_model(
                    X, y, fs_func, classifier, n_splits=5, return_metrics=True, **fs_kwargs
                )
                print_selected_features(selected_features, selected_feature_names, print_feat=False)

                result["selected_features"] = selected_features
                result["selected_feature_names"] = list(selected_feature_names) if selected_feature_names is not None else []
                result["metrics"] = avg_metrics
                result["fold_metrics"] = fold_metrics

            elif mode == "train":
                # Single run on training data → classify
                if fs_name == "Permutation":
                    select_features = X_clustered
                elif fs_name == "forwards SFS":
                    select_features = X_mRMR
                elif fs_name == "backwards SFS":
                    select_features = X_mRMR
                else:
                    select_features = None  # fallback

                selected_features = failsafe_feature_selection(
                    fs_func, X_train, y_train, min_features=20, classifier=classifier, select_features=select_features, **fs_kwargs
                )

                selected_feature_names = classify(
                    X_train, X_test, y_train, y_test, selected_features, classifier, performance=True
                )
                print_selected_features(selected_features, selected_feature_names, print_feat=False)

                # Prepare data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                X_train_sel = X_train_scaled[:, selected_features]
                X_test_sel = X_test_scaled[:, selected_features]

                if classifier == "SVM":
                    model = cl.applySVM(X_train_sel, y_train)
                elif classifier == "RandomForest":
                    model = cl.applyRandForest(X_train_sel, y_train)
                elif classifier == "LogR":
                    model = cl.applyLogR(X_train_sel, y_train)
                elif classifier == "LDA":
                    model = cl.applyLDA(X_train_sel, y_train)
                elif classifier == "KNN":
                    model = cl.applyKNN(X_train_sel, y_train)

                y_pred = model.predict(X_test_sel)
                try:
                    y_proba = model.predict_proba(X_test_sel)[:, 1]
                except:
                    y_proba = None

                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                except:
                    auc = None

                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                result["selected_features"] = selected_features
                result["selected_feature_names"] = list(selected_feature_names) if selected_feature_names is not None else []
                result["metrics"] = {
                    "num feat": len(selected_features),
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auroc": auc,
                    "sensitivity": sensitivity
                }

            # Save result after each feature selection run
            elapsed = time.time() - start_time
            result["elapsed_seconds"] = elapsed
            save_results(classifier, fs_name, inf_method, result)

        print(f"\n✅ Finished {classifier} pipeline!\n")

def gather_and_rank_results(result_dir="results_graph_total_multisite", metric="f1_score", top_n=5):
    results = []

    for clf in classifiers_to_run:
        for inf in inf_methods:
            path = os.path.join(result_dir, clf, inf, "*.json")
            for file in glob.glob(path):
                with open(file, "r") as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    results.append({
                        "classifier": clf,
                        "inf_method": inf,
                        "fs_method": data.get("feature_selection", "Unknown"),
                        "metric_value": metrics.get(metric, 0),
                    })

    df = pd.DataFrame(results)
    top_results = df.sort_values(by="metric_value", ascending=False).head(top_n)
    print("\n🏆 Top configurations based on", metric)
    print(top_results)

    return top_results

def print_selected_features_from_top_result(result_dir="results_graph_total_multisite", metric="f1_score"):
    top_results = gather_and_rank_results(result_dir=result_dir, metric=metric, top_n=1)
    if top_results.empty:
        print("⚠️ No top result found.")
        return

    top = top_results.iloc[0]
    clf = top["classifier"]
    inf = top["inf_method"]
    fs = top["fs_method"]

    path = os.path.join(result_dir, clf, inf, f"{fs}_*.json")
    best_file = max(glob.glob(path), key=os.path.getctime)  # get the most recent

    with open(best_file, "r") as f:
        data = json.load(f)
        selected_feature_names = data.get("selected_feature_names", [])

    print(f"\n🧠 Best configuration: {clf} + {inf} + {fs}")
    print(f"📂 Loaded from: {best_file}")
    print(f"🔬 Selected Features ({len(selected_feature_names)}):")
    for feat in selected_feature_names:
        print(f" - {feat}")


# ========== PARALLEL RUNNER ========== #

if __name__ == "__main__":
    print("🚀 Starting parallel pipeline...")

    # Run all classifiers in parallel
    Parallel(n_jobs=len(classifiers_to_run))(
        delayed(main_for_classifier)(clf) for clf in classifiers_to_run
    )

    print("\n🎉 All classifiers completed!")

    #top_configs = gather_and_rank_results()
    print_selected_features_from_top_result()

    print("\n🎉 Program finished")
