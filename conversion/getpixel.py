import cv2
import glob
import os
import re
import pickle
import pandas as pd


def main():
    # set mulit-process number, equal to the number of cores
    # process_number = 1
    filepath = input("Filepath: ")
    output_array = []
    for f in glob.glob(filepath + '/*.*'):
        image = cv2.imread(f)
        output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = output.flatten()
        if len(output) != 9216:
            print(f)
        head, tail = os.path.split(f)
        photo_id = re.findall(r'\d+', tail)[0]
        try:
            emotion = re.findall(r'[ \-](\w+)', tail)[0]
        except:
            print(f)
            emotion = ''
        output_array.append([photo_id, output, emotion, tail])

    output_pd = pd.DataFrame(output_array, columns =['id', 'pixels', 'emotion', 'original_file'])

    with open('pixel.pd', 'wb') as fout:
        pickle.dump(output_pd, fout)

if __name__ == "__main__":
    main()