import pickle
from sklearn.model_selection import train_test_split


with open('models/vectorizer.pkl',"rb") as f:
    vectorizer = pickle.load(f) 

with open('models/X_Tfidf.pkl',"rb") as f:
    X = pickle.load(f)     

with open('models/y.pkl',"rb") as f:
    y = pickle.load(f)   

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

pickle.dump(model, open('models/trained_model.pkl', 'wb'))


