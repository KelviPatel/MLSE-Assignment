import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml
import joblib

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

path=config['training']['source_dir']
output=config['training']['output']
x_test_path=config['training']['x_test']
y_train_path=config['training']['y_test']

n_estimators = params['training']['n_estimators']
max_depth = params['training']['max_depth']

df=pd.read_csv(path)

X=df.drop(columns=['Stress_Level'])
y=df['Stress_Level']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)

x1=pd.DataFrame(X_test)
y1=pd.DataFrame(y_test)

x1.to_csv(x_test_path)
y1.to_csv(y_train_path)

# print(X_train,y_train)


model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
model.fit(X_train,y_train)

joblib.dump(model, output)


