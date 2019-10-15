import pandas
import numpy
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt


#method for prediting, plotting and evaluation report generation
def PredictNPlot(testSet1=None,trainSet1=None,attributes=None,Predictor=None):
    lstt = []

    lstt.append('Prediction with attributes : %s' %str(attributes))

    lm1 = LinearRegression()
    lm1.fit(trainSet1[attributes].get_values(), trainSet1[Predictor].get_values())
    lstt.append('intercept = %f'%lm1.intercept_)
    lstt.append('coefficent : [ %s ]' %str(lm1.coef_))
    # print the coefficients
    print(lm1.intercept_)
    print(lm1.coef_)

    #predict
    y_pred = lm1.predict(testSet1[attributes].get_values())
    lst=[]
    woy=pandas.DataFrame(testSet1['weekoftheyear'].get_values().tolist(),columns=['weekoftheyear'],dtype=float)
    lst.append(woy)
    tc=pandas.DataFrame(testSet1['total_calls'].get_values().tolist(),columns=['total_calls'],dtype=float)
    lst.append(tc)
    y_predDF=pandas.DataFrame(y_pred,columns=['predicted_total_calls'],dtype=float)
    lst.append(y_predDF)

    pdata=pandas.concat(lst,axis=1)
    fif, ax = plt.subplots()
    #ax.set_ylim(-0.2, 1.2)
    #ax.set_xlim(-0.2, 1.2)
    RMSE=numpy.sqrt(metrics.mean_squared_error(testSet1['total_calls'].get_values(), y_pred))
    MAE=metrics.mean_absolute_error(testSet1['total_calls'].get_values(), y_pred)
    MSE=metrics.mean_squared_error(testSet1['total_calls'].get_values(), y_pred)
    lstt.append('Root Mean Square Error: %f'%RMSE)
    lstt.append('Mean Absolute Error: %f' %MAE)
    lstt.append('Mean Square Error: %f' % MSE)
    pdata.boxplot(by=['weekoftheyear'], ax=ax)
    fif.savefig("[%s].png" %str(attributes))
    lstt.append('------XXXXXX--------')
    ls=[]
    for dfi in testSet1.columns:
        xi=pandas.DataFrame(testSet1[dfi].get_values(),columns=[dfi])
        ls.append(xi)
    ls.append(y_predDF)
    test=pandas.concat(ls,axis=1)
    gdata1 = sns.pairplot(test, hue="Quarterinhour",
                          x_vars=attributes,
                          y_vars=['predicted_total_calls'])
    gdata1._legend.remove()
    handles = gdata1._legend_data.values()
    labels = gdata1._legend_data.keys()
    gdata1.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4)
    gdata1.savefig("PairPlotWithCalls%s.png"%str(attributes))
    return lstt


def dataTransformer(path=None, delimiter=','):
    data = pandas.read_csv(path, sep=delimiter)
    data.replace(to_replace='?', value=numpy.nan, inplace=True)
    # data = data.replace(to_replace=[-1], value=[numpy.nan], inplace=True)
    #data.dropna(inplace=True)
    lst = []
    # changing the columns to type category
    for cols in data.columns:
        # data[cols].replace(to_replace=['?'], value=[" "], inplace=True)
        if data[cols].dtype.name == 'object':
            #working with time and extracting month day and week related informations
            if cols=="TimeInstant" or cols=="timestamp" or cols=="date_time":
                xt=pandas.to_datetime(data[cols],errors='coerce')
                lst.append(xt)
                wd=xt.dt.dayofweek # day of week
                mth=xt.dt.month# month of year
                hr=xt.dt.hour#hour of the day
                #qtr=xt.dt.quarter #quarter of the year
                date=xt.dt.date #date
                dim=xt.dt.daysinmonth#daysinmonth
                minutes=xt.dt.minute#minutesinhour
                week=xt.dt.week#weekoftheyear
                dof=xt.dt.dayofyear#dayofyear
                dof1=pandas.DataFrame(dof.get_values(),columns=['Dayofyear'],dtype=mth.dtype)
                month=pandas.DataFrame(mth.get_values(),columns=['Month'],dtype=mth.dtype)
                hour=pandas.DataFrame(hr.get_values(),columns=['Hours'],dtype=hr.dtype)
                weekday=pandas.DataFrame(wd.get_values(),columns=['Weekday'],dtype=wd.dtype)
                date1 = pandas.DataFrame(date.get_values(), columns=['Date'])
                daysinmonth = pandas.DataFrame(dim.get_values(), columns=['Dayinmonth'], dtype=wd.dtype)
                minutes1 = pandas.DataFrame(minutes.get_values(), columns=['Quarterinhour'], dtype=wd.dtype)
                minutes1.replace(to_replace=[0, 15, 30, 45], value=[1, 2, 3, 4], inplace=True)#converting the minutes into quarters
                week1 = pandas.DataFrame(week.get_values(), columns=['weekoftheyear'], dtype=wd.dtype)
                lst.append(date1)
                lst.append(dof1)
                lst.append(week1)
                lst.append(month)
                lst.append(daysinmonth)
                lst.append(weekday)
                lst.append(hour)
                lst.append(minutes1)
            else:
                lst.append(data[cols].astype('category'))
        else:
            lst.append(data[cols].astype(float))
    data1=pandas.concat(lst, axis=1)
    data1x=data1[['Month','weekoftheyear','Dayofyear','Dayinmonth', 'Weekday', 'Hours', 'Quarterinhour', 'total_calls']].copy()
    #Pair plot for days in month
    '''
    #uncomment for plotting the graph
    
    gdata1 = sns.pairplot(data1x, hue="Quarterinhour",
                     x_vars=['Month', 'weekoftheyear','Dayofyear', 'Dayinmonth', 'Weekday', 'Hours', 'Quarterinhour'],
                     y_vars=['total_calls'])
    gdata1._legend.remove()
    handles=gdata1._legend_data.values()
    labels = gdata1._legend_data.keys()
    gdata1.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4)
    gdata1.savefig("PairPlotWithCalls.png")
    '''
    #-------------------------------------------------------------------------------
    #Heat Map of data
    '''
    #uncomment for plotting the graph
    
    ax = sns.heatmap(data1x.corr())
    ax.figure.savefig("heatmap.png")
    '''
    #-------------------------------------------------------------------------------
    #Pairwise Linear regression and plot
    '''
    #uncomment for plotting the graph
    
    reg = sns.pairplot(data1x, x_vars=['Month', 'weekoftheyear','Dayofyear' , 'Dayinmonth', 'Weekday', 'Hours', 'Quarterinhour'],
                       y_vars='total_calls', size=7, aspect=0.7, kind='reg')#using seaborn for pairwise linear regresion analyasis
    reg.savefig('LinearRegressionFitPairwise.png')
    '''
    # -------------------------------------------------------------------------------
    #Correlation-plot
    '''
    #uncomment for plotting the graph
    
    gdata2 = sns.pairplot(data1x, hue="Quarterinhour")
    gdata2._legend.remove()
    handles = gdata2._legend_data.values()
    labels = gdata2._legend_data.keys()
    gdata2.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4)
    gdata2.savefig("FeaturePairPlot.png")
    '''
    datax = data1.sort_values(by=['Month'])#sorting the data according to month
    dtest, dtrain = [x for _, x in datax.groupby(datax['weekoftheyear'] < 47)]

    testSet1 = dtest[['weekoftheyear','Dayinmonth', 'Weekday', 'Hours', 'Quarterinhour', 'total_calls']].copy()#Selecting the features for test
    trainSet1 = dtrain[['weekoftheyear','Dayinmonth', 'Weekday', 'Hours', 'Quarterinhour', 'total_calls']].copy()#selecting the features for training


    lstt=PredictNPlot(testSet1=testSet1,trainSet1=trainSet1,attributes=['Dayinmonth', 'Weekday', 'Hours', 'Quarterinhour']
                 ,Predictor='total_calls')
    lst1 = PredictNPlot(testSet1=testSet1, trainSet1=trainSet1,
                        attributes=['Weekday', 'Hours', 'Quarterinhour']
                        , Predictor='total_calls')
    lstt = lstt + lst1
    lst1 = PredictNPlot(testSet1=testSet1, trainSet1=trainSet1,
                        attributes=['Weekday', 'Hours']
                        , Predictor='total_calls')
    lstt = lstt + lst1
    lst1 = PredictNPlot(testSet1=testSet1, trainSet1=trainSet1,
                        attributes=['Hours', 'Quarterinhour']
                        , Predictor='total_calls')
    lstt = lstt + lst1
    lst1 = PredictNPlot(testSet1=testSet1, trainSet1=trainSet1,
                        attributes=['Hours']
                        , Predictor='total_calls')
    lstt = lstt + lst1
    lst1 = PredictNPlot(testSet1=testSet1, trainSet1=trainSet1,
                        attributes=['Dayinmonth', 'Hours']
                        , Predictor='total_calls')
    lstt = lstt + lst1


    numpy.savetxt("Report.txt",lstt, delimiter="\n", fmt="%s")

if __name__ == '__main__':

    path='sample_Data.csv'
    dataTransformer(path=path)