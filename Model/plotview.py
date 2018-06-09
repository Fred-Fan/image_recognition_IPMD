import brewer2mpl
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from Utils import load_pd_data
from keras.models import load_model


'''
The method creates visualizations of model predictions on test set. It provides class distriubtions,
images with prediciton results represented by bar charts, confusion matrix and acurracy precision matrix.
@param
X: features of test set
y_prob: predicted probability vectors for all test images, with size num_images * 7. Ex: ((0.16, 0.05 ....), (0.07, 0.34,...), ...... )
y_pred: predicted label for all test images, size num_images * 1 Ex: (1, 4, 2, 0, ....)
y_true: true labels for all test images, size num_images * 1 Ex: (1, 2, 2, 3, .....)
labels: names of classes in list of length 7
plot_wrong_prediction: provides the option to plot only wrongly predicted images

'''
def plot_subjects(start, end, X, y_pre, y_tru, labels, filename, title=False, color=True):
    fig = plt.figure(figsize=(16, 16))
    emotion = dict([[i, labels[i]] for i in range(len(labels))])
    iter = end - start + 1
    for i in range(iter):
    #for i in range(start, end + 1):   
        try:
            input_img = X[start+i, :]
        except IndexError:
            break
        #ax = fig.add_subplot(10, 7, i + 1)
        ax = fig.add_subplot(10, 4, i+1)
        #ax.imshow(input_img.reshape(96,96), cmap=matplotlib.cm.gray, origin = 'lower')
        ax.imshow(input_img.reshape(96,96), cmap=matplotlib.cm.gray)
        pos1 = ax.get_position()
        #print(pos1.x0, pos1.y0,  pos1.width, pos1.height)
        pos2 = [pos1.x0, pos1.y0,  pos1.width*3, pos1.height*3] 
        ax.set_position(pos2)


        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        if y_tru == None:
            plt.title(filename[start+i],fontsize=14)
            plt.xlabel(labels[y_pre[start+i]], fontsize=12)
        elif y_tru != None and color:
            if y_pre[start+i] != y_tru[start+i]:
                plt.title(filename[start+i], color='#53b3cb',fontsize=14,)
                plt.xlabel(labels[y_tru[start+i]], color='#53b3cb', fontsize=12)
                #plt.title('['+ str(start+i) +']' + filename[start+i], color='#53b3cb',fontsize=14,y=1.08)
            else:
                plt.title(filename[start+i],fontsize=14)
                plt.xlabel(labels[y_tru[start+i]], fontsize=12)
                #plt.title('['+ str(start+i) +']' + filename[start+i],fontsize=14, y=1.08)
        else:
            plt.title(filename[start+i],fontsize=14)
        #if title:
            #plt.title(labels[y_pre[i]], color='blue')
        #plt.tight_layout()
    plt.show()

def plot_probs(start, end, X, y_pro, labels):
    fig = plt.figure(figsize=(13, 13))
    iter = end - start + 1
    for i in range(iter):
    #for i in range(start, end + 1):
        try:
            input_img = X[start+i, :]
        except IndexError:
            break
        #ax = fig.add_subplot(10, 7, i + 1)
        ax = fig.add_subplot(10, 4, i + 1)
        ax.bar(np.arange(0, len(labels)), y_pro[start+i], color=set3, alpha=0.5, edgecolor='black', linewidth=0.5)
        for index in range(len(labels)):
            ax.text(index - 0.4 , y_pro[start+i][index] + 0.1, 
                    '{:.2f}'.format(y_pro[start+i][index]),
                    color='red', fontsize=8)

        ax.set_xticks(np.arange(0, len(labels), 1))
        ax.set_xticklabels(labels, rotation=90, fontsize=10)
        ax.set_yticks(np.arange(0.0, 1.1, 0.5))
        plt.tight_layout()
    plt.show()
    
def text_probs(start, end, X, y_pro, labels):
    fig = plt.figure(figsize=(13, 13))
    iter = end - start + 1
    for i in range(iter):
    #for i in range(start, end + 1):
        try:
            input_img = X[start+i, :]
        except IndexError:
            break
        #ax = fig.add_subplot(10, 7, i + 1)
        ax = fig.add_subplot(10, 4, i + 1)
        predict_index = np.argmax(y_pro[start+i])
        for index in range(len(labels)):
            if index != predict_index:
                ax.text(1,(index - 0.4)/2,
                    '{}:{:.2f}'.format(labels[index], y_pro[start+i][index]),
                    color='black', fontsize=12)
            else:
                ax.text(1,(index - 0.4)/2,
                    '{}:{:.2f}'.format(labels[index], y_pro[start+i][index]),
                    color='red', fontsize=12)                
        ax.set_xticks(np.arange(0, len(labels), 1))
        ax.axis('off')
        plt.tight_layout()
    plt.show()

def plot_subjects_with_probs(start, end, X, y_pro, y_pre, y_tru, labels, filename):
    #iter = int((end - start) / 7)
    iter = int((end - start) / 4)
    for i in range(iter):
        #plot_subjects(i * 7, (i + 1) * 7 - 1, X, y_pre, y_tru, labels, filename, title=False)
        #plot_probs(i * 7, (i + 1) * 7 - 1, X, y_pro, labels)
        #plot_subjects(i * 4, (i + 1) * 4 - 1, X, y_pre, y_tru, labels, filename, title=False)
        #plot_probs(i * 4, (i + 1) * 4 - 1, X, y_pro, labels)
        plot_subjects(start+i * 4, start+(i + 1) * 4 - 1, X, y_pre, y_tru, labels, filename, title=False)
        plot_probs(start+i * 4, start+(i + 1) * 4 - 1, X, y_pro, labels)

def plot_distribution(y_true, y_pred, labels):
    ind = np.arange(1, len(labels) + 1, 1)  # the x locations for the groups
    width = 0.4
    fig, ax = plt.subplots()
    true = ax.bar(ind, np.bincount(y_true) / len(y_true) * 100, width, color=set3, alpha=1.0, edgecolor='black',
                  linewidth=1)
    pred = ax.bar(ind + width, np.bincount(y_pred) / len(y_pred) * 100, width, color= set3, alpha=0.3,
                  edgecolor='black', linewidth=1)
    ax.set_xticks(np.arange(1, len(labels) + 1, 1))
    ax.set_xticklabels(labels, rotation=30, fontsize=14)
    ax.set_xlim([1.5 * width, len(labels) + 1 - 0.5 * width])
    ax.set_title('True and Predicted Label Count')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(2)
    fig = plt.figure(figsize=(len(labels) + 1, len(labels) + 1))
    matplotlib.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(111)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(matrix)
    thresh = 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i, j] * 100)) + "%", va='center', ha='center',
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title('Confusion Matrix')
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def class_accuracy(y_true, y_pred, emotion):
    cm = confusion_matrix(y_true, y_pred)
    i = [i for i, label in enumerate(labels) if label == emotion][0]
    tp = cm[i, i]
    fn = sum([cm[i, j] for j in range(len(labels)) if j != i])
    fp = sum([cm[j, i] for j in range(len(labels)) if j != i])
    tn = sum([cm[i, j] for j in range(len(labels)) for i in range(0, len(labels) - 1)]) - (tp + fp + fn)
    return float(tp + tn) / sum([tp, fn, fp, tn])
    
def run_visualization(X, y_prob, y_pred, y_true, labels, filename, plot_wrong_prediction=False, raw_start=0, number=36):
    # Prediction Results

    if plot_wrong_prediction:  # plot probability with input image of only wrong prediction
        idx = [i for i in range(len(y_pred)) if y_pred[i] != y_true[i]]
        tempx = X[idx, :]
        tempy = [y_prob[i] for i in idx]
        temppred = [y_pred[i] for i in idx]
        temptrue = [y_true[i] for i in idx]
        tempfile = [filename[i] for i in idx]
        plot_subjects_with_probs(raw_start, raw_start+number, tempx, tempy, temppred, temptrue, labels, tempfile)
    else:
        plot_distribution(y_true, y_pred, labels)  # plot true and predicted value distribution
        plot_subjects_with_probs(raw_start, raw_start+number, X, y_prob, y_pred, y_true, labels, filename)  # plot probability with input image
        print("\n\n       Accuracy of Training\n")
        for emotion in labels:  # print accuracy per classification
            print("%10s   acc = %f" % (emotion, class_accuracy(y_true, y_pred, emotion)))
        print("\n" + classification_report(y_true, y_pred, target_names=labels))
        plot_confusion_matrix(y_true, y_pred, labels, cmap=plt.cm.YlGnBu)
        
        
def run_visualization_prediction(X, y_prob, y_pred, y_true, labels, filename, raw_start=0, number=36):
    # Prediction Results
    if number % 4 != 0:
        iter =  int(number / 4) + 1
    else:
        iter = int(number / 4)
    for i in range(iter):
        plot_subjects(raw_start+i * 4, raw_start+(i + 1) * 4 - 1, X, y_pred, y_true, 
                      labels, filename, title=False, color=False)
        text_probs(raw_start+i * 4, raw_start+(i + 1) * 4 - 1, X, y_prob, labels)

def plot_one_prediction(X, y_prob, y_pred, y_true, labels, filename):
    imagename = input('image name:')
    
    if imagename in filename:
        fig = plt.figure(figsize=(4, 4))
        index = filename.index(imagename)
        ax = fig.add_subplot(1, 1, 1)
        input_img = X[index, :]
        ax.imshow(input_img.reshape(96,96), cmap=matplotlib.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.title(imagename,fontsize=14,)
        
        plt.show()
        
        fig = plt.figure(figsize=(4, 1))
        ax = fig.add_subplot(1, 1, 1)
        predict_index = np.argmax(y_prob[index])
        for i in range(len(labels)):
            if i != predict_index:
                ax.text(1,(i - 0.4)/2,
                    '{}:{:.2f}'.format(labels[i], y_prob[index][i]),
                    color='black', fontsize=12)
            else:
                ax.text(1,(i - 0.4)/2,
                    '{}:{:.2f}'.format(labels[i], y_prob[index][i]),
                    color='red', fontsize=12)                
        ax.set_xticks(np.arange(0, len(labels), 1))
        ax.axis('off')
        plt.show()

    else:
        print('could not find the front face')