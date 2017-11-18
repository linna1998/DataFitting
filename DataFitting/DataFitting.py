from __future__ import print_function
import os
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn import preprocessing

DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/Mymark_20171102/'
SINGLE_DATA_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/Mymark_20171102/单峰'
SINGLE_FIG_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/SingleFigure/'
SINGLE_PARA_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/SinglePara/'

DOUBLE_DATA_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/Mymark_20171102/双峰'
DOUBLE_FIG_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/DoubleFigure/'
DOUBLEE_PARA_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/DoublePara/'

PI = 3.1415926

#Normal
def func(x, a, mu, sigma, b):
    return a * (1 / (np.sqrt(2 * PI)) * mu) * np.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)) + b

alist = []
blist = []
mulist = []
sigmalist = []

# Single.  Read in data.
for name in sorted(os.listdir(SINGLE_DATA_DIR)):
    path = os.path.join(SINGLE_DATA_DIR, name)
    title = name
    f = open(path, encoding='UTF-8')
    score = []
    num = []
    isFirst = 1
    for line in f:       
         if isFirst == 0:                          
             if len(line.split('\t')) > 1:
                 if len((line.split('\t', 2))[0]) > 0:
                     score.append(int(line.split('\t', 2)[0]))
                 if len((line.split('\t', 2))[1]) > 0:
                     num.append(int(line.split('\t', 2)[1]))
             if len(line.split(',')) > 1:
                 if len((line.split(',', 2))[0]) > 0:
                     score.append(int(line.split(',', 2)[0]))
                 if len((line.split(',', 2))[1]) > 0:
                     num.append(int(line.split(',', 2)[1]))
         else:
            isFirst = 0         
    f.close()

     #Standardization
    min_score = min(score)
    max_score = max(score)
    min_num = min(num)
    max_num = max(num)   
    for i in range(len(score)):
        score[i] = (score[i] - min_score) / (max_score - min_score)
    for i in range(len(num)):
        num[i] = (num[i] - min_num) / (max_num - min_num)

    
    popt, pcov = curve_fit(func, score, num,
                           bounds=(0, [10. , 10. , 10. , 10.]))                             
    num_vals = [func(i, popt[0], popt[1], popt[2], popt[3]) for i in score]

    alist.append(popt[0])
    mulist.append(popt[1])
    sigmalist.append(popt[2])
    blist.append(popt[3])

    plt.title(title)
    plot1 = plt.plot(score, num, 's', label='original values')
    plot1 = plt.plot(score, num_vals, 'r', label='Normal fit values')
    plt.xlabel('score')
    plt.ylabel('num')

    plt.show()
    plt.savefig(SINGLE_FIG_DIR + title.split('.', 2)[0] + '.png')
    plt.close()

    plt.title(title + '-dAG')       
    plot2 = plt.plot(score, np.sqrt(np.abs(num)) - np.sqrt(np.abs(num_vals)), 'r', label='dAG')   
    plot2 = plt.plot(score, np.zeros(len(score)), 'k', label='Standard')   
    plt.xlabel('score')
    plt.ylabel('dAG num')
    plt.show()
    plt.savefig(SINGLE_FIG_DIR + title.split('.', 2)[0] + '-dAG.png')
    plt.close()

fa = open(os.path.join(SINGLE_PARA_DIR, 'a.txt'),'w+')
fb = open(os.path.join(SINGLE_PARA_DIR, 'b.txt'),'w+')
fmu = open(os.path.join(SINGLE_PARA_DIR, 'mu.txt'),'w+')
fsigma = open(os.path.join(SINGLE_PARA_DIR, 'sigma.txt'),'w+')
for i in alist:
     print(i,file=fa)  
for i in blist:
     print(i,file=fb)  
for i in mulist:
     print(i,file=fmu)  
for i in sigmalist:
     print(i,file=fsigma)  
