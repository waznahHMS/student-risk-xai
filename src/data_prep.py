import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, RISK_THRESHOLD_GRADE


def load_and_prep_data(csv_path=None):
    csv_path = csv_path or DATA_PATH
    df = pd.read_csv(csv_path, sep=';')
    df['at_risk'] = (df['G3'] < RISK_THRESHOLD_GRADE).astype(int)

    X = df.drop(columns=['G1', 'G2', 'G3', 'at_risk'])
    y = df['at_risk']

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    return X_train, X_test, y_train, y_test, preprocessor


def get_feature_names(preprocessor, X_train):
    cat = preprocessor.named_transformers_['cat'] \
        .get_feature_names_out(X_train.select_dtypes(include=['object']).columns)
    num = X_train.select_dtypes(exclude=['object']).columns
    return list(num) + list(cat)
