from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

colonnes_texte = ["job", "marital", "education", "default", "housing",
                   "loan", "contact", "month", "poutcome"]
colonnes_nombres = ["age", "balance", "day_of_week", "duration",
                      "campaign", "pdays", "previous"]

def fill_missing(X):
    X = X.copy()
    X[colonnes_texte] = X[colonnes_texte].fillna("unknown")
    return X

def build_preprocessor():
    return ColumnTransformer([
        ("texte", OneHotEncoder(handle_unknown="ignore"), colonnes_texte),
        ("nombres", StandardScaler(), colonnes_nombres),
    ])