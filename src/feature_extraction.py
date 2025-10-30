import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

import pickle 

df = pd.read_csv('data/processed/cleaned_data.csv')

X = df["Code_text"]
y = df["label"]

model = TfidfVectorizer()

model.fit(X)

X_Tfidf = model.transform(X)

os.makedirs("models", exist_ok=True)

with open('models/vectorizer.pkl',"wb") as f:
    pickle.dump(model,f)

with open('models/X_Tfidf.pkl',"wb") as f:
    pickle.dump(X_Tfidf,f)    

with open('models/y.pkl',"wb") as f:
    pickle.dump(y,f)   