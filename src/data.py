import pandas as pd
from ucimlrepo import fetch_ucirepo

def loading_bank_data():
    dataset = fetch_ucirepo(id=222)

    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]
    print(dataset.metadata)
    print(dataset.variables) 
    return X, y