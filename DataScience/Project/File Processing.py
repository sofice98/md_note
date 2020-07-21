import os
import numpy as np
import pandas as pd


def file2matrix(filename):
    '''
    将文本记录转化为NumPy
    Input:  filename: 文件名

    Output: dataSet, labels
    '''
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()                             # 截掉前后空格和换行
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # 将字符串标签转化为整型标签
        index += 1
    return returnMat, classLabelVector


# 读csv文件
data = pd.read_csv(path)
