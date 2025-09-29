import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# print(f"Pandas - {pd.__version__}, NumPy - {np.__version__}, Sklearn - {sklearn.__version__}")

# Importing data from train-test-split.py. Contains already split into X and y training data
def import_data(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    return X_train, y_train

# training the model using RandomForestClassifier on the train set
def train_model(X_train, y_train):
    # clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    # clf.fit(X_train, y_train)
    # return clf

    continuous_features = ['BMI', 'Age', 'MentHlth', 'PhysHlth', 'GenHlth', 'Income']
    categorical_features = [col for col in X_train.columns if col not in continuous_features + ['Smoker']]

    prepro = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', 'passthrough', categorical_features)    
        ])

    # clf = Pipeline(steps=[
    #     ('preprocessor', prepro),
    #     ('classifier', LGBMClassifier(
    #         random_state=42,
    #         class_weight='balanced',  # can also try "balanced"
    #         n_estimators=300,
    #         learning_rate=0.05,
    #         max_depth=-1,
    #         num_leaves=31))
    # ])

    clf = Pipeline(steps=[
        ('preprocessor', prepro),
        ('classifier', XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=1.25,
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8
        ))
    ])

    clf.fit(X_train, y_train)
    return clf
    
# evaluating the model on the validation set and plotting the ROC curve
def evaluate_model(clf, X_val, y_val):
    val_acc = clf.score(X=X_val, y=y_val)
    print(f"The model's accuracy on the test set is: {val_acc*100}%")
    print(classification_report(y_val, clf.predict(X_val)))

    y_probs = clf.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_probs, pos_label=1)
    auc = roc_auc_score(y_val, y_probs)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    # plt.show()
   

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
