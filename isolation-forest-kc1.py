from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from scipy.io import arff
import pandas as pd
from math import floor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from numpy import mean
from numpy import std
import numpy as np
import code
from matplotlib import pyplot as plt
import seaborn as sns
import os

def get_DB(df):
    df = shuffle(df)
    negative_database = df.loc[df['problems'] <= 0]
    positive_database = df.loc[df['problems'] > 0]
    train_array_sizes = [floor(negative_database.shape[0]*0.3),floor(negative_database.shape[0]*0.4),floor(negative_database.shape[0]*0.5)]

    ### train Base at 30%:

    train_30 = negative_database[0:train_array_sizes[0]]
    test_30 = pd.concat([negative_database[train_array_sizes[0]:],positive_database])
    train_y_30 = train_30['problems'].values
    train_x_30 = train_30.drop(columns=['problems']).values
    test_y_30 = test_30['problems'].values
    test_x_30 = test_30.drop(columns=['problems']).values

    ### train Base at 40%:

    train_40 = negative_database[0:train_array_sizes[1]]
    test_40 = pd.concat([negative_database[train_array_sizes[1]:],positive_database])
    train_y_40 = train_40['problems'].values
    train_x_40 = train_40.drop(columns=['problems']).values
    test_y_40 = test_40['problems'].values
    test_x_40 = test_40.drop(columns=['problems']).values

    ### train Base at 50%:
    train_50 = negative_database[0:train_array_sizes[2]]
    test_50 = pd.concat([negative_database[train_array_sizes[2]:],positive_database])
    train_y_50 = train_50['problems'].values
    train_x_50 = train_50.drop(columns=['problems']).values
    test_y_50 = test_50['problems'].values
    test_x_50 = test_50.drop(columns=['problems']).values
    
    return [(train_x_30,test_x_30,test_y_30,"30%"),(train_x_40,test_x_40,test_y_40,"40%"),(train_x_50,test_x_50,test_y_50,"50%")]

def executeIsolationForest(train_test_list):
    model = IsolationForest(contamination=0.15)
    results_list = []
    
    for i in train_test_list:
        train_x, test_x, test_y, label = i  
        model.fit(train_x)
        yhat = model.predict(test_x)
        test_y[test_y == 1] = -1
        test_y[test_y == 0] = 1
        score = f1_score(test_y,yhat,pos_label=-1)
        tp, fn, fp, tn = confusion_matrix(test_y,yhat).ravel()
        results_list.append([score,(tn, fp, fn, tp),label])


    return results_list
    
def execute_experiment(df):
    f1_scores_30_list = []
    f1_scores_40_list = []
    f1_scores_50_list = []

    best_case_30 = None
    best_case_40 = None
    best_case_50 = None



    for i in range (0,100):
        train_test_list = get_DB(df)
        result_list = executeIsolationForest(train_test_list)
        for result in result_list:
            f1,values,label = result
            if label == "30%":
                f1_scores_30_list.append(f1)
                if best_case_30 == None:
                    best_case_30 = result
                elif best_case_30[0] < f1:
                    best_case_30 = result
            elif label == "40%":
                f1_scores_40_list.append(f1)
                if best_case_40 == None:
                    best_case_40 = result
                elif best_case_40[0] < f1:
                    best_case_40 = result
            else:
                f1_scores_50_list.append(f1)
                if best_case_50 == None:
                    best_case_50 = result
                elif best_case_50[0] < f1:
                    best_case_50 = result
    
    # 30% - Média e desvio padrão do F1
    f1_mean_30, f1_std_30 = mean(f1_scores_30_list), std(f1_scores_30_list)
    print('Training set at 30%: {} F1-score (+/- {})'.format(f1_mean_30,f1_std_30))

    # 40% - Média e desvio padrão do F1
    f1_mean_40, f1_std_40 = mean(f1_scores_40_list), std(f1_scores_40_list)
    print('Training set at 40%: {} F1-score (+/- {})'.format(f1_mean_40,f1_std_40))

    # 50% - Média e desvio padrão do F1
    f1_mean_50, f1_std_50 = mean(f1_scores_50_list), std(f1_scores_50_list)
    print('Training set at 50%: {} F1-score (+/- {})'.format(f1_mean_50,f1_std_50))

    f1_list_list = [f1_scores_30_list,f1_scores_40_list,f1_scores_50_list]

    box_plot_f1_measures(f1_list_list)
    plot_confusion_matrix([best_case_30,best_case_40,best_case_50])

def plot_confusion_matrix(result_list):
    for result in result_list:
        fig = plt.figure()
        cf_matrix = result[1]
        cf_matrix = np.asarray(cf_matrix).reshape(2,2)


        group_names = ['TN','FP','FN','TP']
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1} \n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        plt.title('Train set at ' + result[2], fontsize = 20)

        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',cbar=False)
        plt.xlabel('Predicted Label') 
        plt.ylabel('True Label') 

        plt.savefig("confusion_matrix_" + result[2] + "-KC1.png")
        print("Confusion MATRIX and F1-Score best case for Training set at {}: Matrix (TN,FP,FN,TP) = {}, F1-Score = {}".format(result[2],result[1],result[0]))


def box_plot_f1_measures(f1_results):
    fig, ax = plt.subplots()
    ax.set_xlabel('Training_sets')
    ax.set_ylabel('F1-score')
    ax.boxplot(f1_results)
    ax.set_xticklabels(["30%","40%","50%"], fontsize=8)
    fig.savefig('F1-Score-Boxplot-KC1.png', bbox_inches='tight')

data = arff.loadarff("kc1.arff")
df = pd.DataFrame(data[0])
df.rename(columns = {'defects': 'problems'}, inplace = True)
df['problems'] = df['problems'].apply(lambda x: x.decode("utf-8"))
df['problems'] = df['problems'].map({"false": 0, "true": 1})
df['problems']
execute_experiment(df)
