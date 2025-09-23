import pandas as pd
import numpy as npls
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# print(pd.__version__)
# print(np.__version__)
# print(plt.__version__)
# print(sklearn.__version__)

heart_raw = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
# print(heart_raw.head())

X = heart_raw.drop("HeartDiseaseorAttack", axis=1)

# Create y (the target column)
y = heart_raw["HeartDiseaseorAttack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)