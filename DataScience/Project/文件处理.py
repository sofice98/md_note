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
        returnMat[index, :] = listFromLine[0:3]         # 前几列作为特征
        classLabelVector.append(int(listFromLine[-1]))  # 最后一列数字作为标签
        index += 1
    return returnMat


# 读csv文件
data = pd.read_csv(path)
