from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import pickle

with open('models/trained_model.pkl',"rb") as f:
    classify = pickle.load(f) 

with open('models/X_Tfidf.pkl',"rb") as f:
    X = pickle.load(f)     

with open('models/y.pkl',"rb") as f:
    y = pickle.load(f)   

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


y_pred = classify.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

with open('models/evaluation_report.txt', 'w') as f:
    f.write("Accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n")
    f.write("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)) + "\n")
    f.write("Classification Report:\n" + classification_report(y_test, y_pred))
