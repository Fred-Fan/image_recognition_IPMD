import multiprocessing
from multiprocessing import Pool
import cv2
import glob
import argparse
import os


def resize(inputlist):
    """
    This function is to resize images.

    Args:
        inputlist[0](str): The input path of the image.
        inputlist[1](str): The output path of the image.

    Returns:
        None
        save the resized image into the output directory

    """
    head, tail = os.path.split(inputlist[0])
    image = cv2.imread(inputlist[0])
    #r = 96.0 / image.shape[1]
    #dim = (96, int(image.shape[0] * r))
    dim = (96,96)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    filepath = os.path.join(inputlist[1], tail)
    cv2.imwrite(filepath, resized)
    print(filepath)


def cropped(inputlist):
    """
    This function is to crop images.

    Args:
        inputlist[0](str): The input path of the image.
        inputlist[1](str): The output path of the image.

    Returns:
        None
        save the cropped image into the output directory

    """
    head, tail = os.path.split(inputlist[0])
    image = cv2.imread(inputlist[0])
    # 723*512
    cropped = image[65:483, 162:580]
    # 640*480
    #cropped = image[60:426, 146:512]
    filepath = os.path.join(inputlist[1], tail)
    cv2.imwrite(filepath, cropped)
    print(filepath)


def main():
    """
    This function is to take the command line and then proccess the images.
    
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="Path to the directory that contains the images.")
    ap.add_argument("-m", "--mode", required=True,
                    help="choose mode, 1 for resize image, 2 for crop image")
    ap.add_argument("-o", "--output", required=True,
                    help="Path to the directory to save the modified images. If it does not exist, it will be created.")
    ap.add_argument("-p", "--process", default=1, type=int,
                    help="The number of processes to run simultaneously, the default is 1.")
    args = ap.parse_args()

    inputpath = args.input
    outputpath = args.output
    process_number = args.process

    if not os.path.isdir(inputpath):
        print("Input directory does not exist.")
        os._exit(0)
    inputdir = os.path.join(inputpath, '*.*')

    if not os.path.isdir(outputpath):
        print("Creating {}".format(outputpath))
        os.mkdir(outputpath)

    pool = Pool(processes=process_number)

    if args.mode == '1':
        #resize all the images whose path name matches the 'inputdir'.
        results = [pool.apply_async(resize, ([f, outputpath],)) for f in glob.glob(inputdir)]
    elif args.mode == '2':
        #crop all the images whose path name matches the 'inputdir'.
        results = [pool.apply_async(cropped, ([f, outputpath],)) for f in glob.glob(inputdir)]
    else:
        print("Wrong mode.")
        os._exit(0)

    [r.get() for r in results]
    pool.close()
    pool.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run for windows
    main()
