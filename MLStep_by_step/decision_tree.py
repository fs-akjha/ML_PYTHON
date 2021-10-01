import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima=pd.read_csv(r"C:\Akash Files\ML_PYTHON\MLStep_by_step\pima-indians-diabetes.csv", header=None, names=col_names)
print(pima.head())
features_cols=['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X=pima[features_cols]
y=pima.label
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
result=confusion_matrix(y_test,y_pred)
print("Confusion Matrix: ",result)
result1=classification_report(y_test,y_pred)
print("Classification Report: ",result1)
result2=accuracy_score(y_test,y_pred)
print("Accuracy Score: ",result2)