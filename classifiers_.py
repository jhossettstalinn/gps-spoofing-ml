import sys
import argparse
import pandas as pd
import numpy as np
import preprocess as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# ML Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def cli_flag():
    parser = argparse.ArgumentParser(
        description="Run classifiers with optional feature selection")
    parser.add_argument("--fs",
        choices=["spearman", "lasso", "none"],
        default="none",
        help="Select the feature selection method."
        )
    parser.add_argument("--gs", choices=["yes", "none"],default="none",)
    parser.add_argument("--model", 
                        choices=["kNN", "Nu-SVM", "GNB", "BNB", "ANN", "all"], 
                        default="all")
    args = parser.parse_args()
    return args


def run_gsearch(model_name, clf, X_train_best, y_train, X_test_best):

    param_grids = {
    'kNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]     # Manhattan, Euclidean
    },
    'Nu-SVM': {
        'nu': [0.1, 0.25, 0.5],
        'kernel': ['rbf', 'poly'], 
        #'degree': [2, 3],
        'gamma': ['scale', 'auto'],
        #'tol': [1e-5, 1e-4]
    },
    'GNB': {
        'var_smoothing': np.logspace(-12, -6, 7)
    },
    'BNB': {
        'alpha': [5, 10, 15],
        'binarize': [0.22, 0.25, 0.28],
        'fit_prior': [True, False]
    },
    'ANN': {
        #'hidden_layer_sizes': [(16,), (16, 16), (20,), (20, 20)], 
        'hidden_layer_sizes': [(50, 50), (100, 50)], #(50,), (100,), 
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [1e-4, 1e-3, 1e-2],  # L2 regularization
        'learning_rate_init': [1e-3, 1e-4]
    }
    }

    grid = param_grids[model_name]

    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=grid,
        cv=10,
        scoring='accuracy',      
        n_jobs=-1,               
        verbose=2
    )

    print(f"Training {model_name} model...")
    gsearch.fit(X_train_best, y_train)

    with open("outputs/Evaluation.txt", "a") as f:
        f.write("="*54 + "\n")
        f.write(f"{model_name} - BEST PARAMETERS\n")
        f.write("="*54 + "\n")
        f.write(str(gsearch.best_params_) + "\n")

    print("Testing model...")
    best_model = gsearch.best_estimator_
    y_pred = best_model.predict(X_test_best)

    return y_pred


if __name__ == "__main__":

    args = cli_flag()

    RANDOM_STATE = 12

    try:
        ds = pd.read_csv("outputs/ds_sampled.csv", header=0)
        X = ds.drop(columns=['Output'])
        y = ds['Output']

    except FileNotFoundError:
        ds_original = pd.read_csv(
            "data/dataset_0.csv", header=0)

        # Balancing and sampling the dataset
        ds = pp.sampleDS(ds_original, RANDOM_STATE, N=10000, e=0.5)
        X = ds.drop(columns=['Output'])
        y = ds['Output']

        print("Dataset sampled and saved as 'ds_sampled.csv'")

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=2/3, 
        random_state=RANDOM_STATE, 
        stratify=y
        )

    # Normalizing the features using Min-Max technique
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    with open("outputs/Evaluation.txt", "w") as f:
        f.write("")

    X_train_s = pd.DataFrame(X_train_s, columns=X.columns)
    X_test_s = pd.DataFrame(X_test_s, columns=X.columns)

    # Selecting the feature selection method
    if args.fs == "spearman":

        threshold = 0.95
        S_matrix = pp.spearmancorr(X_train_s)
        to_drop = pp.feature_to_drop(S_matrix, threshold=threshold)

        with open("outputs/Evaluation.txt", "a") as f:
            f.write("="*54 + "\n")
            f.write(f"Spearman threshold: {threshold}\n")
            f.write(f"Features to drop: {to_drop}\n")
        print("Spearman correlation matrix saved\n")

        # Dropping the "worst" features
        X_train_best = X_train_s.drop(columns=list(to_drop))
        X_test_best = X_test_s.drop(columns=list(to_drop))

    elif args.fs == "lasso":

        to_drop = pp.lasso(X_train_s, y_train, RANDOM_STATE)
        print(to_drop)     

        with open("outputs/Evaluation.txt", "a") as f:
            f.write("="*54 + "\n")
            f.write(f"Features to drop: {to_drop}\n")

        # Dropping the "worst" features
        X_train_best = X_train_s.drop(columns=list(to_drop))
        X_test_best = X_test_s.drop(columns=list(to_drop))

    else:                                  # No feature selection
        X_train_best = X_train_s
        X_test_best = X_test_s

    # Classifiers with best hyperparameter by default
    classifiers = [
        ("kNN", KNeighborsClassifier(n_neighbors=5,
                                     p=1, #Manhattan
                                     weights='distance')),
        ("Nu-SVM", NuSVC(nu=0.25,
                         kernel='rbf',
                         gamma='scale',
                         random_state=RANDOM_STATE)),
        ("GNB", GaussianNB(var_smoothing=1e-12)),
        ("BNB", BernoulliNB(alpha=5,
                            binarize=0.28,
                            fit_prior=True)),
        ("ANN", MLPClassifier(activation='tanh',
                              alpha=1e-4,
                              hidden_layer_sizes=(100, 50),
                              learning_rate_init=0.001,
                              max_iter=800,
                              tol=1e-4, #default
                              random_state=RANDOM_STATE))
    ]

    for model_name, clf in classifiers:
        
        if model_name != args.model and args.model != "all":
            continue
        else:
            pass

        if args.gs == "yes":
            y_pred = run_gsearch(model_name, clf, 
                                X_train_best, y_train, X_test_best)
        else:
            print(f"Training {model_name} model...")
            clf.fit(X_train_best, y_train)
            
            print("Testing model...")
            y_pred = clf.predict(X_test_best)

        
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        TN, FP, FN, TP = cm.ravel()
        acc = accuracy_score(y_test, y_pred) * 100
        TPR = TP / (TP + FN) * 100
        FPR = FP / (FP + TN) * 100
        FNR = FN / (TP + FN) * 100
        print("Model successfully evaluated.\n")

        with open("outputs/Evaluation.txt", "a") as f:
            f.write("="*54 + "\n")
            f.write(f"{model_name} - CLASSIFICATION REPORT\n")
            f.write("="*54 + "\n")
            f.write(report)
            f.write("\n")
            f.write(f"Accuracy: {acc:.2f}\n")
            f.write(f"Probability of Detection: {TPR:.2f}\n")
            f.write(f"Probability of False Alarm: {FPR:.2f}\n")
            f.write(f"Probability of Misdetection: {FNR:.2f}\n")
            f.write("\n")
            f.write("CONFUSION MATRIX\n")
            f.write("="*54 + "\n")
            f.write(np.array2string(cm, separator='  '))
            f.write("\n\n")

