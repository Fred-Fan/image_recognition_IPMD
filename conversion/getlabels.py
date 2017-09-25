import glob
import pandas as pd
import re
from collections import Counter
import os

label_list = []
id_list = []
for f in glob.glob("D:\\Downloads\\kaggle\\*.*"):
    try:
        head, tail = os.path.split(f)
        file_id = int(re.findall(r'\d+', tail)[0]) - 1
        emotion = re.findall(r'\-(\w+)\-', f)[0]
        label_list.append([file_id, emotion])
        id_list.append(file_id)
    except:
        print(f)

# show missed id
true_set = set(list(range(min(id_list), max(id_list)+1)))
actual_set = set(id_list)
print('missed', true_set - actual_set)

# show duplicated id and remove
duplicate_list = [item for item, count in Counter(id_list).items() if count > 1]
print('duplicated', duplicate_list)

label_pd = pd.DataFrame(label_list, columns = ['id', 'emotion'])

for i in duplicate_list:
    label_pd = label_pd[label_pd['id'] != i]

label_pd.to_csv('correct_label.csv')