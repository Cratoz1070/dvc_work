import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_SETS = {
    "baseline": (
        ["Pclass", "Age", "SibSp", "Parch", "Fare"],
        ["Sex", "Embarked"],
    ),
    "extended": (
        ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
         "HasCabin", "FarePerPerson", "Child", "Mother"],
        ["Sex", "Embarked", "Title", "FamilyCategory", "Deck"],
    ),
}


def add_features(data):
    result = data.copy()
    title = result["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    result["Title"] = title.where(title.isin({"Mr", "Mrs", "Miss", "Master"}), "Other")
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    result["FamilyCategory"] = pd.cut(
        result["FamilySize"], [0, 1, 4, 7, np.inf],
        labels=["Alone", "Small", "Medium", "Large"],
    )
    result["HasCabin"] = result["Cabin"].notna().astype(int)
    result["Deck"] = result["Cabin"].str[0].fillna("Unknown")
    result["FarePerPerson"] = result["Fare"] / result["FamilySize"]
    return result


class FillAge(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_ = X["Age"].median()
        return self

    def transform(self, X):
        result = X.copy()
        result["Age"] = result["Age"].fillna(self.median_)
        result["Child"] = (result["Age"] < 14).astype(int)
        result["Mother"] = (
            (result["Sex"] == "female")
            & (result["Age"] >= 18)
            & (result["Parch"] > 0)
            & (result["Title"] != "Miss")
        ).astype(int)
        return result


def make_pipeline(numeric, categorical, forest, seed):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("age", FillAge()),
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=seed, **forest)),
    ])


def main():
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    data = pd.read_csv("data/train.csv")
    y = data["Survived"]
    X = add_features(data.drop(columns="Survived"))
    numeric, categorical = FEATURE_SETS[params["features"]]
    features = numeric + categorical
    seed = params["cv"]["random_state"]
    cv = StratifiedKFold(n_splits=params["cv"]["folds"], shuffle=True, random_state=seed)
    pipeline = make_pipeline(numeric, categorical, params["forest"], seed)

    predictions = []
    scores = []
    importance = []
    for fold, (fit_idx, check_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(pipeline).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        check = X.iloc[check_idx]
        actual = y.iloc[check_idx]
        predicted = model.predict(check)
        probability = model.predict_proba(check)[:, 1]
        predictions.append(pd.DataFrame({
            "passenger_id": check["PassengerId"].to_numpy(), "fold": fold,
            "actual": actual.to_numpy(), "predicted": predicted, "probability": probability,
        }))
        scores.append({
            "accuracy": accuracy_score(actual, predicted),
            "f1": f1_score(actual, predicted, zero_division=0),
            "roc_auc": roc_auc_score(actual, probability),
        })
        prepared = model.named_steps["age"].transform(check)
        result = permutation_importance(
            model[1:], prepared[features], actual, scoring="accuracy",
            n_repeats=5, random_state=seed,
        )
        importance.append(result.importances_mean)

    output = Path("evaluation")
    output.mkdir(exist_ok=True)
    metrics = pd.DataFrame(scores).mean().to_dict()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    oof = pd.concat(predictions, ignore_index=True).sort_values("passenger_id")
    oof.to_csv(output / "predictions.csv", index=False)
    fpr, tpr, _ = roc_curve(oof["actual"], oof["probability"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(output / "roc.csv", index=False)
    pd.DataFrame({"feature": features, "accuracy_drop": np.mean(importance, axis=0)}).to_csv(
        output / "importance.csv", index=False,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
