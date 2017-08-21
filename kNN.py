# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:43:22 2017

@author: User
"""

import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#k近邻算法
#inX是用于分类的输入向量，dataSet是训练样本集，labels是标签向量，k表示用于选择最近邻的数目
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #计算距离
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet    #计算inX和dataSet中每个点的差异
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5                        #差异的平方和开根号，即欧氏距离
    print(type(distances))
    #argsort将distances中的元素从小到大排列，提取其对应的索引
    sortedDistIndicies = distances.argsort()            
    classCount = {}                                     #创建字典保存预测类别和次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #get(voteIlabel,0)中的0表示当键votIlabel不在字典时，其值默认为0
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #iteritems迭代返回字典中的所有项，key设置按照哪一列来排序，0表示按键排序，1表示按值排序
#    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),\
#                              reverse = True)  
    #dict.iteritems()为python2的用法,python3中为dict.items()
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),\
                              reverse = True)  
    return sortedClassCount[0][0]

#准备数据：从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)                #得到文件行数
    #创建返回的特征矩阵，行数和文件行数一样，列为3，即有3个特征
    returnMat = np.zeros((numberOfLines,3)) 
    classLabelVector = []                           #标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        #选取前三个元素作为特征，保存在特征矩阵中
        returnMat[index,:] = listFromLine[0:3]
        #选取最后一个元素作为标签，保存到标签向量中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#准备数据：归一化数值
#有多种归一化方法，可以使用归一化公式newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    #min中的参数0使得函数可以从列中选取最小值，而不是从行中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值减最小值，即公式中的分母
    ranges = maxVals - minVals              
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]                    #m为数据集的行数
    #当前值减去最小值即公式中的分子
    #minVals和ranges的值都是1*3,使用numpy.tilt函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = dataSet - np.tile(minVals,(m,1)) 
    #具体特征值相除，即公式。注意对于某些数值处理软件包，/可能意味着矩阵除法，但是在numpy中，
    #矩阵除法需要使用函数linalog.solve(matA,matB)     
    normDataSet = normDataSet/np.tile(ranges,(m,1))     
    return normDataSet,ranges,minVals

#测试算法：作为完整程序验证分类器
def datingClassTest():
    #设置测试集占10%的数据
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)        #归一化
    m = normMat.shape[0]                                    # m为normMat的行数
    numTestVecs = int(m*hoRatio)                            #测试样本的个数
    errorCount = 0                                          #计数器变量保存预测错误的个数
    for i in range(numTestVecs):
        #前numTestVecs为测试样本集，第i个测试样本为normMat[i,:],第numTestVecs+1个到最后一个样本为
        #训练样本集normMat[numTestVecs:m,:],同理datingLabels[numTestVecs:m]为对应的标签向量
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                    datingLabels[numTestVecs:m],3)
        print('the classifier came back with: %s, the real answer is: %s'\
              %(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('the total error rate is: %f' %(errorCount/float(numTestVecs)))

#使用算法：构建完整可用系统
def classifyPerson():
    resultList = ['not at all','in samll doses','in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per pear?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person: ',resultList[classifierResult-1])


