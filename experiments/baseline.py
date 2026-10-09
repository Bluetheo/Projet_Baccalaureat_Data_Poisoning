from src.split import split_data
from src.data import loading_bank_data
from src.preprocessing import build_preprocessor, fill_missing
from src.models import get_logistic_regression, get_random_forest
from src.evaluate import evaluate_model
import pandas as pd
from pathlib import Path


X, y = loading_bank_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

X_train = fill_missing(X_train)
X_val = fill_missing(X_val)
X_test = fill_missing(X_test)

preprocessor = build_preprocessor()
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)
X_test_prep = preprocessor.transform(X_test)

print("Shape train :", X_train_prep.shape)
print("Shape val   :", X_val_prep.shape)
print("Shape test  :", X_test_prep.shape)

models = {
    "logistic_regression": get_logistic_regression(),
    "random_forest": get_random_forest(),
}

for name, model in models.items():
    model.fit(X_train_prep, y_train)
    metrics = evaluate_model(model, X_test_prep, y_test)
    print(f"\n{name} :")
    for key, value in metrics.items():
        print(f"  {key} : {value}")

rows = []
for name, model in models.items():
    model.fit(X_train_prep, y_train)
    metrics = evaluate_model(model, X_test_prep, y_test)
    metrics["model"] = name
    rows.append(metrics)
    print(f"\n{name} :")
    for key, value in metrics.items():
        print(f"  {key} : {value}")

results_df = pd.DataFrame(rows)
output_path = Path("results/csv/baseline.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\nRésultats sauvegardés dans {output_path}")