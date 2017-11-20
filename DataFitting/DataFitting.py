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
DOUBLE_PARA_DIR = 'L:/学术/大三/语言统计分析/高考分数段单双峰统计/DoublePara/'

PI = 3.1415926

# Normal, single
def func(x, a, mu, sigma):
    return a * (1 / (np.sqrt(2 * PI)) * mu) * np.exp(-(x - mu) * (x - mu) / (2
    * sigma * sigma))

## Normal, double
#def func(x, a1, mu1, sigma1, a2, mu2, sigma2):
#    return a1 * (1 / (np.sqrt(2 * PI)) * mu1) * np.exp(-(x - mu1) * (x - mu1) / (2 * sigma1 * sigma1)) + a2 * (1 / (np.sqrt(2 * PI)) * mu2) * np.exp(-(x - mu2) * (x - mu2) / (2 * sigma2 * sigma2)) 


alist1 = []
mulist1 = []
sigmalist1 = []
alist2 = []
mulist2 = []
sigmalist2 = []

# Single.  Read in data.
for name in sorted(os.listdir(SINGLE_DATA_DIR)):
    path = os.path.join(SINGLE_DATA_DIR, name)
## Double.  Read in data.
#for name in sorted(os.listdir(DOUBLE_DATA_DIR)):
#    path = os.path.join(DOUBLE_DATA_DIR, name)
    title = name
    f = open(path, encoding='UTF-8')
    score = []
    num = []
    isFirst = 1
    for line in f:       
         if isFirst == 0:                          
             if len(line.split('\t')) > 1:
                 if len((line.split('\t', 2))[0]) > 0:
                     score.append(float(line.split('\t', 2)[0]))
                 if len((line.split('\t', 2))[1]) > 0:
                     num.append(int(line.split('\t', 2)[1]))
             if len(line.split(',')) > 1:
                 if len((line.split(',', 2))[0]) > 0:
                     score.append(float(line.split(',', 2)[0]))
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

    ## Double
    #popt, pcov = curve_fit(func, score, num,
    #                       bounds=(0, [10. , 10. , 10. , 10. ,10. ,10.]))                             
    #num_vals = [func(i, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) for i in score]
    #if popt[0]>popt[3]:
    #    alist1.append(popt[0])
    #    mulist1.append(popt[1])
    #    sigmalist1.append(popt[2])
    #    alist2.append(popt[3])
    #    mulist2.append(popt[4])
    #    sigmalist2.append(popt[5])
    #else:
    #    alist1.append(popt[3])
    #    mulist1.append(popt[4])
    #    sigmalist1.append(popt[5])
    #    alist2.append(popt[0])
    #    mulist2.append(popt[1])
    #    sigmalist2.append(popt[2])

     # Single
    popt, pcov = curve_fit(func, score, num,
                           bounds=(0, [10.  , 10.  , 10.  ]))
    num_vals = [func(i, popt[0], popt[1], popt[2]) for i in score]
    alist1.append(popt[0])
    mulist1.append(popt[1])
    sigmalist1.append(popt[2])    

    plt.title(title)
    plot1 = plt.plot(score, num, 's', label='original values')
    plot1 = plt.plot(score, num_vals, 'r', label='Normal fit values')
    plt.xlabel('score')
    plt.ylabel('num')

    plt.show()
    #plt.savefig(DOUBLE_FIG_DIR + title.split('.', 2)[0] + '.png')
    plt.savefig(SINGLE_FIG_DIR + title.split('.', 2)[0] + '.png')
    plt.close()

    plt.title(title + '-dAG')       
    plot2 = plt.plot(score, np.sqrt(np.abs(num)) - np.sqrt(np.abs(num_vals)), 'r', label='dAG')   
    plot2 = plt.plot(score, np.zeros(len(score)), 'k', label='Standard')   
    plt.xlabel('score')
    plt.ylabel('dAG num')
    plt.show()
    #plt.savefig(DOUBLE_FIG_DIR + title.split('.', 2)[0] + '-dAG.png')
    plt.savefig(SINGLE_FIG_DIR + title.split('.', 2)[0] + '-dAG.png')
    plt.close()

fa1 = open(os.path.join(SINGLE_PARA_DIR, 'a.txt'),'w+')
fmu1 = open(os.path.join(SINGLE_PARA_DIR, 'mu.txt'),'w+')
fsigma1 = open(os.path.join(SINGLE_PARA_DIR, 'sigma.txt'),'w+')
#fa1 = open(os.path.join(DOUBLE_PARA_DIR, 'a1.txt'),'w+')
#fmu1 = open(os.path.join(DOUBLE_PARA_DIR, 'mu1.txt'),'w+')
#fsigma1 = open(os.path.join(DOUBLE_PARA_DIR, 'sigma1.txt'),'w+')
#fa2 = open(os.path.join(DOUBLE_PARA_DIR, 'a2.txt'),'w+')
#fmu2 = open(os.path.join(DOUBLE_PARA_DIR, 'mu2.txt'),'w+')
#fsigma2 = open(os.path.join(DOUBLE_PARA_DIR, 'sigma2.txt'),'w+')

for i in alist1:
     print(i,file=fa1)  
for i in mulist1:
     print(i,file=fmu1)  
for i in sigmalist1:
     print(i,file=fsigma1)  
#for i in alist2:
#     print(i,file=fa2)  
#for i in mulist2:
#     print(i,file=fmu2)  
#for i in sigmalist2:
#     print(i,file=fsigma2)  