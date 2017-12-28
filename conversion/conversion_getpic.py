import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool
import pickle
import os
import cv2


def savepic(inputlist):
    #target = np.asarray(np.split(np.asarray(inputlist[0][1]), 48))
    target = np.asarray(inputlist[1]).reshape(96, 96)
    # print(image)
    print(inputlist[2], os.getpid())
    filename = "images/" + str(inputlist[0]) + "-" + str(inputlist[2])
    cv2.imwrite(filename,target)
    #plt.imshow(target, cmap='gray', interpolation="bicubic", vmin=0, vmax=255)
    #plt.savefig(filename, dpi=113)
    # change the output folder according to OS
    # plt.show()


def main():
    #newStart = 1  # the picture number you want to start with
    process_number = 8
    file = input('where is the file:')
    pool = Pool(processes=process_number)
    with open(file, 'rb') as fin:
        imagelist = pickle.load(fin)
    #for i in range(len(imagelist)):
    for i in range(len(imagelist)):
        temp = imagelist.loc[i]
        if temp.pixels is not None:
            inputlist = [temp.id, temp.pixels, temp.original_file]
            savepic(inputlist)
        else:
            print(temp.original_file, 'no pixels')

    #end_num = len(imageList)
    # end_num = 22001
    #print(newStart, end_num)
    #for num in range(newStart - 1, end_num):
        #inputlist = [imageList[num], num]
        #result = pool.apply_async(savepic, (inputlist,))
        #result.get()
    #pool.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
