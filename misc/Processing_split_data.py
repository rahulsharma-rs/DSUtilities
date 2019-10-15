''' This file a '''
import numpy
from UtilsRAN import processInputCSVFile
from sklearn.model_selection import KFold
import RAN_kfold as RAN
def processData(data=None,labeled=True):
    #croping label from the data
    if labeled==True:
        dta=data.T[:-1].T
    elif labeled==False:
        dta=data
    else:
        print ("wrong value for label")
        return None, None, None, None, None

    #sorting the data in ascenting order of label
    sortedData = numpy.asarray(sorted(data,key=lambda test:test[test.shape[0]-1]))
    # croping label from the train data
    if labeled==True:
        label=sortedData.T[sortedData.T.shape[0]-1].T
        # identifying unique labels in sorted order and their frequency
        uLabels, labelFreq = numpy.unique(label, return_counts=True)
    else:
        label, uLabels, labelFreq=None,None,None

    #cropping the sorted data
    sortedData = sortedData.T[:-1].T
    #returning the
    return dta, sortedData, label, uLabels, labelFreq


def main():
    r=[]
    path = "/home/rahul/Dropbox/RahulSharma/PHD/DataSets/Data_cortex_Nuclear/mice_with_class_label.csv"
    rawData = processInputCSVFile(path)
    labels=rawData.T[rawData.T.__len__()-1].T
    #data_train, data_test, labels_train, labels_test = train_test_split(rawData, labels, test_size=0.20, random_state=42)
    kf = KFold(n_splits=10)
    for train, test in kf.split(rawData):
        ran=None
        ran = RAN.RAN()
        X_train, X_test, Y_train, Y_test = rawData[train], rawData[test], labels[train], labels[test]
        trainD,testD,label,unique_label, labelFrequency=processData(train=X_train,test=X_test)
        # --------------------------------------------------------------------------------------------------------
        ran.rawData=trainD #initializing RAN with training data
        ran.intermediateData=[]# initializing the intermediated data
        ran.intermediateData.append(ran.rawData)
        # training
        ran.train_RAN_Upward(ran)
        # ran.intermediateData1= ran.intermediateData
        ran.dataProjection = [None] * (ran.numberOfLayers)
        ran.intermediateTestData = [None] * (ran.numberOfLayers)
        # temp=[]
        ran.actualActivation = [[]] * (ran.numberOfLayers)
        for dt in ran.intermediateData:
            ran.startPoint.append(numpy.random.rand(dt.shape[1]))

        tempw = []
        if ran.weight[0].min() < 0 or ran.weight[0].max() > 1:
            # ran.weight[0]= 2*ran.sigmoid(ran.weight[0])-1
            for x in ran.weight[0]:
                if x.min() < 0 or x.max() > 1:
                    tempw.append(ran.normalizeData(x))
                else:
                    tempw.append(x)
            ran.weight[0] = numpy.asarray(tempw)
            # ran.weight[0]=ran.normalizeData(ran.weight[0])
        # --------------------------------------------------------------------------------------------------------
        for w in ran.weight:
            ran.weightDown.append(numpy.transpose(w))

            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            # validating by upward propagation
        #path_test = "/home/rahul/Dropbox/RahulSharma/PHD/DataSets/Gas_Sensor_data_for_home_activity_monitoring/reduced_train1_data.csv"
        #ran.testData = processInputCSVFile(path_test)
        ran.testData(testD)
        ran.intermediateData1.append(ran.testData)
        ran.propagateUp(ran)

            # --------------------------------------------------------------------------------------------------------
        print ("x")
        r.append(ran)

    print ("x")


if __name__ == "__main__":
    main()