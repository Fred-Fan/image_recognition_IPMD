import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool
import pickle
import time


def savepic(inputlist):
    target = np.asarray(np.split(np.asarray(inputlist[0][1]), 48))
    # print(image)
    print(inputlist[1]+1)
    if (inputlist[1]+1) % 100 == 0:
        print(time.time())
    plt.imshow(target, cmap='gray', interpolation="bicubic", vmin=0, vmax=255)
    filename = "images/No." + str(inputlist[1] + 1) + "-" + inputlist[0][0] + "-" + inputlist[0][2] + ".png"
    plt.savefig(filename, dpi=113)
    # change the output folder according to OS
    # plt.show()


def main():
    newStart = 25000  # the picture number you want to start with
    process_number = 8
    start_time = time.time()

    pool = Pool(processes=process_number)
    with open('imagelist.pkl', 'rb') as fin:
        imageList = pickle.load(fin)
    #end_num = len(imageList)
    end_num = 30000
    print(newStart, end_num)
    #for num in range(newStart - 1, end_num):
        # inputlist = [imageList[num], num]
    result = [pool.apply_async(savepic, ([imageList[num], num],)) for num in range(newStart - 1, end_num)]
    [r.get() for r in result]
    #if num % 100 == 0:
    #    print(time.time()-start_time)
    pool.close()
    pool.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
