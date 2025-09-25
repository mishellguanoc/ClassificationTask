import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ===== 1. Cargar dataset =====
data_path = "data/bank-full.csv"   # asegúrate que el dataset esté en la carpeta data
df = pd.read_csv(data_path, sep=";")

# ===== 2. Definir features y target =====
X = df.drop("y", axis=1)   # "y" es la columna target en el Bank Marketing dataset
y = df["y"]

# ===== 3. Identificar tipos de variables =====
categorical_features = X.select_dtypes(include=["object"]).columns
numeric_features = X.select_dtypes(exclude=["object"]).columns

# ===== 4. Pipelines de preprocesamiento =====
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numeric_transformer, numeric_features)
    ]
)

# ===== 5. Crear pipeline con modelo =====
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# ===== 6. Train/test split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===== 7. Entrenar =====
clf.fit(X_train, y_train)

# ===== 8. Evaluar =====
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===== 9. Guardar modelo =====
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/decision_tree_pipeline.pkl")
print("✅ Modelo guardado en models/decision_tree_pipeline.pkl")
