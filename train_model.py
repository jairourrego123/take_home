import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt

def train_model(data_path, model_path):
    # Load Data
    df = pd.read_csv(data_path)
    df_X = df.drop("y", axis=1)
    df_label = df["y"]

    numeric_features = ["x1", "x2", "x4", "x5"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["x3", "x6", "x7"]
    categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", LogisticRegression(max_iter=10000))]
    )

    RANDOM_STATE = 1337

    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_label, random_state=RANDOM_STATE
    )

    clf.fit(X_train, y_train)
    print("Model score: %.3f" % clf.score(X_test, y_test))
    
    tprobs = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, clf.predict(X_test)))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, clf.predict(X_test)))
    print(f'AUC: {roc_auc_score(y_test, tprobs)}')

    RocCurveDisplay.from_estimator(estimator=clf, X=X_test, y=y_test)
    plt.savefig('roc_curve.png')
    
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()
    train_model(args.data_path, args.model_path)
