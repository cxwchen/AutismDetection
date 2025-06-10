import json
import os
import time
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from classification.src import classifiers as cl
from featureselection.src.feature_selection_methods import *
from Pipeline import load_full_corr, train_and_evaluate, cross_validate_model, print_selected_features, failsafe_feature_selection, classify, load_dataframe
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from featureselection.src import cluster

# ========== CONFIGURATION ========== #

classifiers_to_run = ["SVM", "RandomForest", "LogR", "LDA", "KNN"]

feature_selection_methods = [
    #("Lasso_selection", Lasso_selection, {"alpha": 0.044984, "max_iter": 2000}, "cv"), #0.044984 for full corr
    #("HSIC_Lasso", hsiclasso, {"num_feat": 19}, "cv"), #98 for full corr
    #("mRMR", mRMR, {"num_features_to_select": 100}, "cv"),
    #("Permutation", Perm_importance, {}, "train"),
    ("forwards SFS", forwards_SFS, {"n_features_to_select": 20}, "train"),
    #("backward SFS", backwards_SFS, {"n_features_to_select": 10}, "train")
]
#("ReliefF", reliefF_, {"num_features_to_select": 200}, "cv")
# ========== SAVE RESULTS ========== #

def save_results(classifier, feature_selection_name, results_dict):
    os.makedirs(f"results/{classifier}", exist_ok=True)
    
    # Generate timestamp: YYYYMMDD-HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filename = f"results/{classifier}/{feature_selection_name}_{timestamp}.json"
    
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
    
    # Load data
    X, y = load_full_corr()

    # Run raw model just for reference
    X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

    # Use clustered features for Perm / SFS
    X_clustered = cluster.cluster(X_train, y_train, t=2)
    X_mRMR = mRMR(X_train, y_train, classifier, num_features_to_select=70)

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
            selected_features, selected_feature_names, avg_metrics = cross_validate_model(
                X, y, fs_func, classifier, n_splits=5, return_metrics=True, **fs_kwargs
            )
            print_selected_features(selected_features, selected_feature_names, print_feat=False)

            result["selected_features"] = selected_features
            result["selected_feature_names"] = list(selected_feature_names) if selected_feature_names is not None else []
            result["metrics"] = avg_metrics

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
        save_results(classifier, fs_name, result)

    print(f"\n✅ Finished {classifier} pipeline!\n")

# ========== PARALLEL RUNNER ========== #

if __name__ == "__main__":
    print("🚀 Starting parallel pipeline...")

    # Run all classifiers in parallel
    Parallel(n_jobs=len(classifiers_to_run))(
        delayed(main_for_classifier)(clf) for clf in classifiers_to_run
    )

    print("\n🎉 All classifiers completed!")