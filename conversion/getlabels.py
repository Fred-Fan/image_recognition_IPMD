import glob
import pandas as pd
import re

label_list = []
for f in glob.glob("kaggle\*.*"):
    try:
        file_id = re.findall(r'No\.(\d+)', f)[0]
        emotion = re.findall(r'\-(\w+)\-', f)[0]
        label_list.append([file_id, emotion])
    except:
        print(f)

label_pd = pd.DataFrame(label_list)

label_pd.to_csv('correct_label.csv')