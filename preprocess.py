from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


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
    result["Cabin"] = result["Cabin"].fillna("Unknown")
    result["FarePerPerson"] = result["Fare"] / result["FamilySize"]
    return result


def prepare(data, age_median):
    result = add_features(data)
    result["Age"] = result["Age"].fillna(age_median)
    result["Fare"] = result["Fare"].fillna(result["Fare"].median())
    result["FarePerPerson"] = result["FarePerPerson"].fillna(result["FarePerPerson"].median())
    result["Embarked"] = result["Embarked"].fillna(result["Embarked"].mode().iloc[0])
    result["Child"] = (result["Age"] < 14).astype(int)
    result["Mother"] = (
        (result["Sex"] == "female")
        & (result["Age"] >= 18)
        & (result["Parch"] > 0)
        & (result["Title"] != "Miss")
    ).astype(int)
    return result


def main():
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    raw = pd.read_csv("data/raw.csv")
    train, test = train_test_split(
        raw,
        test_size=params["preprocess"]["test_size"],
        random_state=params["preprocess"]["random_state"],
        stratify=raw["Survived"],
    )
    age_median = train["Age"].median()
    prepare(train, age_median).to_csv("data/train.csv", index=False)
    prepare(test, age_median).to_csv("data/test.csv", index=False)


if __name__ == "__main__":
    main()
