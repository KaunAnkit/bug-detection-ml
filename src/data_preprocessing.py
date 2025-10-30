import os
import pandas as pd

label =[]
codes = []


bug_list = os.listdir("data/raw/bug")
correct_list = os.listdir("data/raw/correct")


for x in range(len(os.listdir("data/raw/bug"))):

    path = os.path.join("data/raw/bug", bug_list[x])


    with open(path, "r") as file:
        text = file.read()

        codes.append(text)
        label.append(0)

for x in range(len(os.listdir("data/raw/correct"))):

    path = os.path.join("data/raw/correct", correct_list[x])


    with open(path, "r") as file:
        text = file.read()

        codes.append(text)
        label.append(1)



df = pd.DataFrame({
    "Code_text" : codes,
    "label" : label
})

df.to_csv("data/processed/cleaned_data.csv", index=False)
