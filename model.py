import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

model_path = "model.pkl"
dataset_path = "dataset"
label_path = "labels.pkl"

X=[]
y=[]

for file in os.listdir(dataset_path):
    if file.endswith(".npy"):
        ges_name=file.replace("npy","")
        path = os.path.join(dataset_path,file)

        data = np.load(path)

        for row in data:
            X.append(row)
            y.append(ges_name)

X=np.array(X,dtype = np.float32)
y=np.array(y)

#print(len(X),X.shape)

le = LabelEncoder()
y_enc=le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y_enc,test_size=0.2,random_state=0,stratify=y_enc)

model = RandomForestClassifier(n_estimators=200, random_state=0)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

c=classification_report(y_test,y_pred,target_names=le.classes_)
print(f"Classification Report : \n{c}")

joblib.dump(model,model_path)
joblib.dump(le,label_path)