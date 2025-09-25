import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

# print(f"Pandas - {pd.__version__}, NumPy - {np.__version__}, Sklearn - {sklearn.__version__}")

# Importing data from train-test-split.py. Contains already split into X and y training data
def import_data(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    return X_train, y_train

# training the model using RandomForestClassifier on the train set
def train_model(X_train, y_train):
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    return clf

# evaluating the model on the validation set andplotting the ROC curve
def evaluate_model(clf, X_val, y_val):
    val_acc = clf.score(X=X_val, y=y_val)
    print(f"The model's accuracy on the test set is: {val_acc*100}%")

    y_probs = clf.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_probs, pos_label=1)
    auc = roc_auc_score(y_val, y_probs)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
   

def main():
    X_train, y_train = import_data("X_train.csv", "y_train.csv")
    
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    clf = train_model(X_train_new, y_train_new)
    
    evaluate_model(clf, X_val, y_val)


if __name__ == "__main__":
    # Measure execution time
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")


    
    # clf = train_model(X_train_new, y_train_new)
    
    # evaluate_model(clf, X_val, y_val)