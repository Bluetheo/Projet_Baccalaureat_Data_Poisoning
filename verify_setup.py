from src.data import loading_bank_data


X, y = loading_bank_data()
print(X.head())
print(X.shape)
print(y.value_counts())
