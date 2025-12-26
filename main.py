!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd

df = pd.read_csv("parkinsons.csv")

print("Shape:", df.shape)
print(df.head())
selected_features = ["MDVP:Fo(Hz)", "PPE"]
target_feature = "status"   # 0/1 classification

X = df[selected_features]
y = df[target_feature]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Validation accuracy:", acc)
import joblib

joblib.dump(model, 'my_model.joblib')

