import csv
import pickle


def get_user_list(filename='test.csv'):
    """return list of user tuple. e.g. ('001,', 'demo1'), ('002,', 'demo2')]"""
    user_list = []
    category_dict = {'0':'Angry', '1':'Disgust', '2':'Fear', '3':'Happy', '4':'Sad', '5':'Surprise', '6':'Neutral'}
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        for row in spamreader:
            try:
                pic_category = category_dict[str(row[0])]
                #print(pic_category)
                pic_type = row[2]
                r = row[1].split(' ')
                pixel_list = [float(x) for x in r]
                user_list.append([pic_category, pixel_list,pic_type])
            except:
                print(row)

    return user_list


def main():
    imageList = get_user_list()
    with open('imagelist.pkl', 'wb') as fout:
        pickle.dump(imageList, fout)


if __name__ == "__main__":
    main()

