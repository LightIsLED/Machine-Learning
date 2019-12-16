from numpy import *
import operator


def createDataSet():
    global group, label
    group = array([[30, 2.0], [40, 6.7], [50, 15.1], [60, 19.6], [70,27.9],
                   [30, 11.3], [40, 19.2], [50, 32.1], [60, 46.9], [70, 64.7],
                   [30, 11.5], [40, 18.1], [50, 28.2], [60, 33.7], [70, 35.4],
                   [30, 14.9], [40, 19.8], [50, 16.3], [60, 15.6], [70, 11.9],
                   [30, 0.2], [40, 0.4], [50, 3.1], [60, 5.4], [70,16.6]])
    
    labels = ['Diabetes', 'Diabetes', 'Diabetes','Diabetes', 'Diabetes',
                  'Hypertenstion', 'Hypertenstion', 'Hypertenstion', 'Hypertenstion', 'Hypertenstion',
                  'Hypercholesterolemia', 'Hypercholesterolemia',  'Hypercholesterolemia',  'Hypercholesterolemia',  'Hypercholesterolemia',
                  'Hypertriglyceridemia', 'Hypertriglyceridemia', 'Hypertriglyceridemia', 'Hypertriglyceridemia', 'Hypertriglyceridemia',
                  'Chronic-Kidney-Diseases',  'Chronic-Kidney-Diseases',  'Chronic-Kidney-Diseases',  'Chronic-Kidney-Diseases',  'Chronic-Kidney-Diseases'] # R: Romance, A: Action


    return labels


def classify(inX, dataSet, labels, k): # classifier method(inX: data that we want to know, dataSet: trained dataset, labels: label of each trained data, k: number of neighbors we chose)

    dataSetSize = dataSet.shape[0] # total data train set's size
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # make matrix that is fully completed with inX, then calc difference
    sqDiffMat = diffMat ** 2 # mult each data in diffMat
    sqDistances = sqDiffMat.sum(axis = 1) # sum each data's mult results in diffMat
    distances = sqDistances ** 0.5 # sqrt sqDistances
    sortedDistIndices = distances.argsort() # sort sqDistances so low data comes to the top
    classCount = {}

    for i in range(k): # select k(3) nearest neighbors from to the top: (0, 1, 2)
        voteIlabel = labels[sortedDistIndices[i]] # find each neighbor's label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # count up neighbor's label count

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) # sort all label counts in descending order

    return sortedClassCount[0][0] # return most voted label


