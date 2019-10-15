
import pandas
import sklearn
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit,ShuffleSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import  Pipeline
from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy
import itertools
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def plotROC(fpr=None,tpr=None,path=None,auc=None):
    plt.figure()

    #ax=plt.subplot(121)
    lw = 2
    #color = ['b', 'g', 'r', 'c', 'k', ]\
    skip=['gold','yellow','ivory','greenyellow','white','floralwhite','aliceblue','ghostwhite','lavander','honeydes','w','whitesmoke','snow','lemonchiffon','azure','linen',
          'antiquewhite','papayawhip','oldlace','cornsilk','palegoldenrod','lightyellow','mintcream','lightcyan','lavenderblush']
    color = []
    for cname, cvalue in matplotlib.colors.cnames.iteritems():
        if cname in skip:
            continue
        else:
            color.append(cname)

    ctr=0
    for i in range(0,len(fpr)):
        plt.plot(fpr[i], tpr[i], color=color[i],
             lw=lw, label='Class-%0.0f (AUC = %0.2f%%)' % (i,auc[i]*100.00))
        ctr+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc=1, borderaxespad=0.,prop={'size':10})
    #lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.savefig(path)

    #plt.show()
    plt.close()
#def test(data,labels):
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,pltname=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontsize=7)
    plt.yticks(tick_marks, classes,rotation=90,fontsize=7)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #



    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=6)
    plt.xlabel('Predicted Label',fontsize=6)

    plt.savefig(pltname)
    plt.close()


def main():
        x = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    #using IRIS data			
    data = df = pd.read_csv('https://raw.github.com/pandas-dev/pandas/master''/pandas/tests/data/iris.csv')
    data.dropna(how='any',inplace=True)

    #dx1=data[data.columns[6:-6]]
    #dx2=data[data.columns[25:-1]]
    #data1=pandas.concat([dx1,dx2],axis=1)
    data1=data[x]
    #dx=data1.apply(stats.zscore)
    #z = numpy.abs(stats.zscore(data1))
    #outlier_index=numpy.unique(numpy.where(z > 3)[0]/x.__len__())
    #data.drop(labels=outlier_index.tolist(),inplace=True)
    raw_data=data1.get_values()
    #data1 = data[x]
    labels=data['Name'].get_values()
    report1 = []
    report_AUC = []
    avPrecision = [[], [], [], [], [],[],[],[],[],[]];
    avRecall = [[], [], [], [], [],[],[],[],[],[]];
    avF1 = [[], [], [], [], [],[],[],[],[],[]];
    avAccuracy = [[], [], [], [], [],[],[],[],[],[]]

    # loop for number of iterations of the experiment
    for topiter in range(0, 10): #loop 30 times
        iter1 = 0
        tr_size = 0.9  # size of train data
        # loop for the 9 Research Designs of the experiment
        for iter1 in range(1, 10):
            r = []
            reports = []
            acc = [[], [], [], [], [],[],[],[],[],[]]
            pr = [[], [], [], [], [],[],[],[],[],[]]
            rec = [[], [], [], [], [],[],[],[],[],[]]
            fone = [[], [], [], [], [],[],[],[],[],[]]
            sup = [[], [], [], [], [],[],[],[],[],[]]
            modelName = []
            pltctr = 0
            fprCurve = []
            tprCurve = []
            fprCurve1 = []
            tprCurve1 = []
            prCurve = []
            rCurve = []
            thresCurve = []
            classLabels = []
            AUC = []
            modelArc = []
            iter = 0
            if tr_size == 0.1:
                print "x"
            #kf2 = ShuffleSplit(n_splits=10, train_size=tr_size, test_size=1 - tr_size, random_state=0)
            kf2 = StratifiedShuffleSplit(n_splits=10, train_size=tr_size, test_size=1 - tr_size, random_state=0)

            tr_size = tr_size - 0.1
            kf = KFold(n_splits=10)
            kf3 = ShuffleSplit(n_splits=10, test_size=0.4, train_size=0.6, random_state=0)
            for train, test in kf2.split(raw_data, labels):
                X_train, X_test, Y_train, Y_test = raw_data[train], raw_data[test], labels[train], labels[test]

                Y_train_temp = Y_train
                Y_test_temp = Y_test
                hotcode=[[1,0,0,0,0],[0,1,0,0,0,],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
                Y2d=[]
                for y2d in Y_test_temp:
                    Y2d.append(hotcode[int(y2d-1)])
                ohc=numpy.asarray(Y2d).transpose()
                reports.append(
                    "----------------------------------Pass %s---------------------------------------------------------------" % iter)

                # -------MLP training
                reports.append("-------------------Multi-layer perceptron model-----------------------")
                modelName.append("MLP")
                print "-------------------------MLP----------------------------------"
                #mlp = MLPClassifier(random_state=1,hidden_layer_sizes=(300,50,150,35,150,100), max_iter=1500,)#77% result
                mlp = MLPClassifier(random_state=1,
                                    hidden_layer_sizes=(50), max_iter=2000, )
                mlp.fit(X_train, Y_train_temp)
                predictionMlp = mlp.predict(X_test)
                print(confusion_matrix(Y_test_temp, predictionMlp))
                print(classification_report(Y_test_temp, predictionMlp, digits=7))
                #fpr2, tpr2, threshold = roc_curve(Y_test_temp, mlp.predict_proba(X_test)[:, 1])

                print "accuracy: %s" % (accuracy_score(Y_test_temp, predictionMlp))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, predictionMlp)))
                reports.append(classification_report(Y_test_temp, predictionMlp, digits=7))
                reports.append("---------------------------------------------------------------------------\n")
                acc[0].append(accuracy_score(Y_test_temp, predictionMlp))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, predictionMlp)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys = yfone
                pr[0].append(numpy.average(yp, weights=ys))
                rec[0].append(numpy.average(yr, weights=ys))
                fone[0].append(numpy.average(yf1, weights=ys))
                sup[0].append(numpy.sum(ys))
                cnf_matrix=confusion_matrix(Y_test_temp, predictionMlp)
                cm_name_path="./Oberservations/Iter%s/pass%s/CM-MLP-fold%s.png" % (topiter, iter1,pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1','Class-2','Class-3','Class-4','Class-5'],
                                      title='confusion matrix',pltname=cm_name_path)

                prob=mlp.predict_proba(X_test)
                prob=prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0,5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-MLP.png" % (topiter, iter1, pltctr), auc=auc)
                # ------------
                # -------------------------------------------------------------
                # logistic regression
                reports.append("----------------------Logistic Regression-------------------------------")
                print "-------------------------Logistic Regression----------------------------------"

                modelName.append("Logistic Regresion")
                logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=5,
                                            class_weight='balanced', C=1)
                mLogreg = logreg.fit(X_train, Y_train_temp)
                pred = mLogreg.predict(X_test)
                print(confusion_matrix(Y_test_temp, pred))
                print(classification_report(Y_test_temp, pred, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pred))

                reports.append("Accuracy: %s" % (accuracy_score(Y_test_temp, pred)))
                reports.append(classification_report(Y_test_temp, pred, digits=7))
                acc[1].append(accuracy_score(Y_test_temp, pred))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pred)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys = yfone
                pr[1].append(numpy.average(yp, weights=ys))
                rec[1].append(numpy.average(yr, weights=ys))
                fone[1].append(numpy.average(yf1, weights=ys))

                sup[1].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, pred)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-LR-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)
                #plt.close()

                prob = logreg.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-LR.png" % (topiter, iter1, pltctr), auc=auc)
                reports.append("--------------------------------------------------------------------------------")
                # -----------------------------------------
                # -------------------------------------------
                ### bernauli rbm
                print "------------------------- LDA----------------------------------"
                reports.append("-------------------- LDA-----------------------------")
                modelName.append("LDA")
                #rbm = BernoulliRBM()
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train, Y_train_temp)
                prd = clf.predict(X_test)

                # with multiple classes
                # print(confusion_matrix((Y_test * 6).astype(int), prd))
                # print(classification_report((Y_test * 6).astype(int), prd))

                # with two classes
                print(confusion_matrix(Y_test_temp, prd))
                print(classification_report(Y_test_temp, prd, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, prd))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, prd)))
                reports.append(classification_report(Y_test_temp, prd, digits=7))
                reports.append("-------------------------------------------------------------------------------------")
                acc[2].append(accuracy_score(Y_test_temp, prd))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, prd)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys= yfone
                pr[2].append(numpy.average(yp, weights=ys))
                rec[2].append(numpy.average(yr, weights=ys))
                fone[2].append(numpy.average(yf1, weights=ys))

                sup[2].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, prd)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-LDA-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)
                #plt.close()

                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-LDA.png" % (topiter, iter1, pltctr), auc=auc)
                # ----------------------------------------------------------------------

                # --------------------------------------------------------------------------
                # K-nearest neighbor
                print "-------------------------K-Neareast Neoghbor----------------------------------"
                # reports.append("--------------------------K-Neareast Neoghbor-------------------------")
                modelName.append("K-NN")
                clf = neighbors.KNeighborsClassifier(
                    n_neighbors=30)  # ,weights='distance',leaf_size=5,algorithm='auto',)
                clf.fit(X_train, Y_train_temp)
                prdt = clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, prdt))
                print(classification_report(Y_test_temp, prdt, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, prdt))
                # reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, prdt)))
                # reports.append(classification_report(Y_test_temp, prdt,digits=7))
                # reports.append("----------------------------------------------------------------------------------------------")
                acc[3].append(accuracy_score(Y_test_temp, prdt))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, prdt)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[3].append(numpy.average(yp, weights=ys))
                rec[3].append(numpy.average(yr, weights=ys))
                fone[3].append(numpy.average(yf1, weights=ys))

                sup[3].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, prdt)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-KNN-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)
                #plt.close()

                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-K-NN.png" % (topiter, iter1, pltctr), auc=auc)
                # -----------------------------------------------------------------------------

                # --------------------------------------------------------------------------
                # SGD classifier
                print "-------------------------SGD Classifier----------------------------------"
                reports.append("-------------------------SGD Classifier---------------------------")
                modelName.append("SGD")
                sgd = SGDClassifier(alpha=0.00001, n_iter=2, epsilon=0.25,loss='modified_huber')
                sgd.fit(X_train, Y_train_temp)
                prdtt = sgd.predict(X_test)
                print(confusion_matrix(Y_test_temp, prdtt))
                print(classification_report(Y_test_temp, prdtt, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, prdtt))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, prdtt)))
                reports.append(classification_report(Y_test_temp, prdtt, digits=7))
                reports.append(
                    "-------------------------------------------------------------------------------------------------------------")
                acc[4].append(accuracy_score(Y_test_temp, prdtt))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, prdtt)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[4].append(numpy.average(yp, weights=ys))
                rec[4].append(numpy.average(yr, weights=ys))
                fone[4].append(numpy.average(yf1, weights=ys))

                sup[4].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, prdtt)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-SGD-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)
                #plt.close()
                prob = sgd.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-SGD.png" % (topiter, iter1, pltctr), auc=auc)

                #pltctr += 1
                # -----------------------------------------------------------------------------
                print "-------------------------Random forest Classifier----------------------------------"
                reports.append("-------------------------Random forest Classifier---------------------------")
                modelName.append("RFC")
                clf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state = 0)
                clf.fit(X_train, Y_train_temp)
                pridct=clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, pridct))
                print(classification_report(Y_test_temp, pridct, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pridct))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, pridct)))
                reports.append(classification_report(Y_test_temp, pridct, digits=7))
                reports.append(
                    "-------------------------------------------------------------------------------------------------------------")
                acc[5].append(accuracy_score(Y_test_temp, pridct))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pridct)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[5].append(numpy.average(yp, weights=ys))
                rec[5].append(numpy.average(yr, weights=ys))
                fone[5].append(numpy.average(yf1, weights=ys))

                sup[5].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, pridct)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-RFC-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)
                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-RFC.png" % (topiter, iter1, pltctr), auc=auc)
                # -----------------------------------------------------------------------------
                print "-------------------------Decision Tree Classifier----------------------------------"
                reports.append("-------------------------Decision Tree Classifier---------------------------")
                modelName.append("DTC")
                clf = DecisionTreeClassifier( random_state=0)
                clf.fit(X_train, Y_train_temp)
                pridct = clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, pridct))
                print(classification_report(Y_test_temp, pridct, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pridct))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, pridct)))
                reports.append(classification_report(Y_test_temp, pridct, digits=7))
                reports.append(
                    "-------------------------------------------------------------------------------------------------------------")
                acc[6].append(accuracy_score(Y_test_temp, pridct))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pridct)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[6].append(numpy.average(yp, weights=ys))
                rec[6].append(numpy.average(yr, weights=ys))
                fone[6].append(numpy.average(yf1, weights=ys))

                sup[6].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, pridct)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-DTC-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)

                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-DTC.png" % (topiter, iter1, pltctr), auc=auc)
                #----------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------
                print "-------------------------Gaussian Naive Bayes----------------------------------"
                reports.append("-------------------------Gaussian Naive Bayes---------------------------")
                modelName.append("GNB")
                clf = GaussianNB()
                clf.fit(X_train, Y_train_temp)
                pridct = clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, pridct))
                print(classification_report(Y_test_temp, pridct, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pridct))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, pridct)))
                reports.append(classification_report(Y_test_temp, pridct, digits=7))
                reports.append(
                    "-------------------------------------------------------------------------------------------------------------")
                acc[7].append(accuracy_score(Y_test_temp, pridct))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pridct)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[7].append(numpy.average(yp, weights=ys))
                rec[7].append(numpy.average(yr, weights=ys))
                fone[7].append(numpy.average(yf1, weights=ys))

                sup[7].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, pridct)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-GNB-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)

                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-GNB.png" % (topiter, iter1, pltctr), auc=auc)
                # ----------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------
                print "-------------------------SVC----------------------------------"
                reports.append("-------------------------SVC---------------------------")
                modelName.append("SVC")
                clf = SVC(gamma=2, C=1,probability=True)
                clf.fit(X_train, Y_train_temp)
                pridct = clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, pridct))
                print(classification_report(Y_test_temp, pridct, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pridct))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, pridct)))
                reports.append(classification_report(Y_test_temp, pridct, digits=7))
                reports.append(
                    "-------------------------------------------------------------------------------------------------------------")
                acc[8].append(accuracy_score(Y_test_temp, pridct))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pridct)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[8].append(numpy.average(yp, weights=ys))
                rec[8].append(numpy.average(yr, weights=ys))
                fone[8].append(numpy.average(yf1, weights=ys))

                sup[8].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, pridct)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-SVC-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)

                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-SVC.png" % (topiter, iter1, pltctr), auc=auc)
                # ----------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------
                print "-------------------------Ada boost Classifier----------------------------------"
                reports.append("-------------------------Ada boost Classifier---------------------------")
                modelName.append("ABC")
                clf = AdaBoostClassifier()
                clf.fit(X_train, Y_train_temp)
                pridct = clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, pridct))
                print(classification_report(Y_test_temp, pridct, digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pridct))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, pridct)))
                reports.append(classification_report(Y_test_temp, pridct, digits=7))
                reports.append(
                    "-------------------------------------------------------------------------------------------------------------")
                acc[9].append(accuracy_score(Y_test_temp, pridct))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pridct)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                # ys =  yfone
                pr[9].append(numpy.average(yp, weights=ys))
                rec[9].append(numpy.average(yr, weights=ys))
                fone[9].append(numpy.average(yf1, weights=ys))

                sup[9].append(numpy.sum(ys))

                cnf_matrix = confusion_matrix(Y_test_temp, pridct)
                cm_name_path = "./Oberservations/Iter%s/pass%s/CM-ABC-fold%s.png" % (topiter, iter1, pltctr)
                plot_confusion_matrix(cnf_matrix, classes=['Class-1', 'Class-2', 'Class-3', 'Class-4', 'Class-5'],
                                      title='confusion matrix', pltname=cm_name_path)

                prob = clf.predict_proba(X_test)
                prob = prob.transpose()
                auc = []
                tp = []
                fp = []
                for cx in range(0, 5):
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(ohc[cx], prob[cx], pos_label=1)
                    auc.append(sklearn.metrics.auc(fpr, tpr))
                    tp.append(tpr)
                    fp.append(fpr)
                plotROC(fpr=fp, tpr=tp,
                        path="./Oberservations/Iter%s/pass%s/ROC-pass-%s-ABC.png" % (topiter, iter1, pltctr), auc=auc)
                # ----------------------------------------------------------------------------------
                pltctr += 1
            reports.append(
                    "Model\taverage precision\tSD-Precision\taverage recall\tSD-Recall\taverage F1\tSD-F1\taverage support\taverage accuracy\tSD-Accuracy")
            for ind in range(0, pr.__len__()):
                averages = "RD-%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
                iter1, modelName[ind], numpy.average(pr[ind]), numpy.std(pr[ind], ddof=1), numpy.average(rec[ind]),
                numpy.std(rec[ind], ddof=1), numpy.average(fone[ind]), numpy.std(fone[ind], ddof=1),
                numpy.average(sup[ind]), numpy.average(acc[ind]), numpy.std(acc[ind], ddof=1))
                reports.append(averages)
                report1.append(averages)

                # saving the report
                avPrecision[ind].append(numpy.average(pr[ind]))
                avRecall[ind].append(numpy.average(rec[ind]))
                avF1[ind].append(numpy.average(fone[ind]))
                avAccuracy[ind].append(numpy.average(acc[ind]))
            numpy.savetxt("./Oberservations/Iter%s/pass%s/Modelcomparison.txt" % (topiter, iter1), reports,
                                  delimiter="\n", fmt="%s")
        report1.append("------------------iteration %s ends------------" % topiter)

        numpy.savetxt(
                "./Oberservations/comparitiveReport.txt",
                report1, delimiter="\n", fmt="%s")


        print "x"
    print "x"

if __name__ == '__main__':
    main()