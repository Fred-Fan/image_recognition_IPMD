import numpy as np
import multiprocessing
from multiprocessing import Pool
import pickle
import os
import cv2

def savepic(inputlist):
    #target = np.asarray(np.split(np.asarray(inputlist[0][1]), 48))
    target = np.asarray(inputlist[0][1]).reshape(48, 48)
    # print(image)
    print(inputlist[1]+1, os.getpid())
    filename = "images/No." + str(inputlist[1] + 1) + "-" + inputlist[0][0] + "-" + inputlist[0][2] + ".png"
    cv2.imwrite(filename,target)
    #plt.imshow(target, cmap='gray', interpolation="bicubic", vmin=0, vmax=255)
    #plt.savefig("images/No." + str(inputlist[1] + 1) + ".png", dpi=113)
    # change the output folder according to OS
    # plt.show()


def main():
    newStart = 1  # the picture number you want to start with
    process_number = 1

    pool = Pool(processes=process_number)
    with open('imagelist.pkl', 'rb') as fin:
        imageList = pickle.load(fin)
    #end_num = len(imageList)
    end_num = 2
    print(newStart, end_num)
    for num in range(newStart - 1, end_num):
        inputlist = [imageList[num], num]
        result = pool.apply_async(savepic, (inputlist,))
        result.get()
    pool.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
