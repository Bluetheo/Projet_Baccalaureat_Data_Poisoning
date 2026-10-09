from sklearn.model_selection import train_test_split
from src.data import loading_bank_data

def split_data(X, y, random_state=42):
    X_train, X_reste, y_train, y_reste = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_reste, y_reste, test_size=0.50, stratify=y_reste, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test