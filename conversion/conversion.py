import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool
import pickle


def savepic(inputlist):
    target = np.asarray(np.split(np.asarray(inputlist[0]), 48))
    # print(image)
    print(inputlist[1])
    plt.imshow(target, cmap='gray', interpolation="bicubic", vmin=0, vmax=255)
    plt.savefig("images/No." + str(inputlist[1] + 1) + ".png", dpi=113)
    # change the output folder according to OS
    # plt.show()


def main():
    newStart = 25000  # the picture number you want to start with
    process_number = 1

    pool = Pool(processes=process_number)
    with open('imagelist.pkl', 'rb') as fin:
        imageList = pickle.load(fin)
    end_num = len(imageList)
    print(newStart, end_num)
    for num in range(newStart - 1, end_num):
        inputlist = [imageList[num], num]
        result = pool.apply_async(savepic, (inputlist,))
        result.get()
    pool.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
