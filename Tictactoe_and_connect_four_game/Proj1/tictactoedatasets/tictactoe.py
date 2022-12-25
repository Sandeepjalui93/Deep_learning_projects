import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import sys
import argparse

def single():
    data =  pd.read_csv('tictac_single.txt', sep=" ", header=None)
    data.columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','y']
    print('The tic tac toe single label dataset is\n',data)

    col=data.columns.tolist()

    #Label encoding
    le = preprocessing.LabelEncoder()

    #Transforming df
    col = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','y']
    for i in range(len(col)):
        data[col[i]]=le.fit_transform(data[col[i]])

    X = data.drop('y', axis = 1)
    y = data['y']
    print('The Unique values of y are',y.unique())
    print('----------------------------------------------------------------------------')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    #Linear SVM
    print('Running Linear SVM')
    clf = OneVsRestClassifier(SVC(kernel='linear',degree=1)).fit(X, y)
    y_pred_svm = clf.predict(X_test)

    print('Accuracy Score for linear SVM',accuracy_score(y_test,y_pred_svm))
    print('Recall Score for linear SVM',recall_score(y_test,y_pred_svm,average='macro'))
    print('Precision Score for linear SVM',precision_score(y_test,y_pred_svm,average='macro'))
    print('F1 Score for linear SVM',f1_score(y_test,y_pred_svm,average='macro'))

    conf_mat_svm = confusion_matrix(y_test, y_pred_svm)
    print('Confusion matrix for linear SVM\n',conf_mat_svm)

    accuracies_clf = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for linear SVM: {:.2f} %".format(accuracies_clf.mean()*100))
    print("Cross validaton Standard Deviation score for linear SVM: {:.2f} %".format(accuracies_clf.std()*100))
    print('----------------------------------------------------------------------------')

    #SVM with kernel RBF
    print('Running SVM with kernel RBF')
    clf = OneVsRestClassifier(SVC(kernel='rbf',degree=1)).fit(X, y)
    y_pred_svm = clf.predict(X_test)

    print('Accuracy Score for RBF SVM',accuracy_score(y_test,y_pred_svm))
    print('Recall Score for RBF SVM',recall_score(y_test,y_pred_svm,average='macro'))
    print('Precision Score for RBF SVM',precision_score(y_test,y_pred_svm,average='macro'))
    print('F1 Score for RBF SVM',f1_score(y_test,y_pred_svm,average='macro'))

    conf_mat_svm = confusion_matrix(y_test, y_pred_svm)
    print('Confusion matrix for RBF SVM\n',conf_mat_svm)

    accuracies_clf = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for RBF SVM: {:.2f} %".format(accuracies_clf.mean()*100))
    print("Cross validaton Standard Deviation score for RBF SVM: {:.2f} %".format(accuracies_clf.std()*100))
    print('----------------------------------------------------------------------------')


    #KNN
    print('Running KNN with N=3')
    knn = make_pipeline(KNeighborsClassifier(n_neighbors=3))
    knn.fit(X_train, y_train)
    y_pred_knn=knn.predict(X_test)

    print('Accuracy Score for KNN',accuracy_score(y_test,y_pred_knn))
    print('Recall Score for KNN',recall_score(y_test,y_pred_knn,average='macro'))
    print('Precision Score for KNN',precision_score(y_test,y_pred_knn,average='macro'))
    print('F1 Score for KNN',f1_score(y_test,y_pred_knn,average='macro'))

    conf_mat_knn = confusion_matrix(y_test, y_pred_knn)
    print('Confusion matrix for KNN\n',conf_mat_knn)

    accuracies_knn = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for KNN: {:.2f} %".format(accuracies_knn.mean()*100))
    print("Cross validaton Standard Deviation score for KNN: {:.2f} %".format(accuracies_knn.std()*100))
    print('----------------------------------------------------------------------------')

    #MLP
    print('Running MLP')
    mlp = make_pipeline(MLPClassifier(hidden_layer_sizes=(2000,1000,500,100,50,), max_iter=200, tol=0.5))
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)

    print('Accuracy Score for MLP',accuracy_score(y_test,y_pred_mlp))
    print('Recall Score for MLP',recall_score(y_test,y_pred_mlp,average='macro'))
    print('Precision Score for MLP',precision_score(y_test,y_pred_mlp,average='macro'))
    print('F1 Score for MLP',f1_score(y_test,y_pred_mlp,average='macro'))

    conf_mat_mlp = confusion_matrix(y_test, y_pred_mlp)
    print('Confusion matrix for MLP\n',conf_mat_mlp)

    accuracies_mlp = cross_val_score(estimator = mlp, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for MLP: {:.2f} %".format(accuracies_mlp.mean()*100))
    print("Cross validaton Standard Deviation score for MLP: {:.2f} %".format(accuracies_mlp.std()*100))
    print('----------------------------------------------------------------------------')
    
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def multi():
    data = np.loadtxt('tictac_multi.txt')
    df = pd.DataFrame(data, columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','y0','y1','y2','y3','y4','y5','y6','y7','y8'])
    print('The tic tac toe multi label dataset is\n',data)

    X = df.iloc[:,0:9]
    y = df.iloc[:,9:]
    print('The Unique values of y0-y8 are',df.iloc[:,9].unique())
    print('----------------------------------------------------------------------------')

    output_arr = ['y0','y1','y2','y3','y4','y5','y6','y7','y8']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    lin_reg = make_pipeline(LinearRegression())
    y_pred_lin = []
    for i in output_arr:
        lin_reg.fit(X_train,y_train[i])
        y_pred_lin.append(lin_reg.predict(X_test))
    final_res = np.array(y_pred_lin)
    
    #Running Linear Regression
    print('Running Linear Regression')
    #Predicting optimal move
    df_pred = pd.DataFrame(final_res.T,columns=output_arr)
    opti = df_pred.idxmax(axis='columns')
    print('The optimal values are\n',opti)
    
    #Calculating accuracy
    accu = np.round(NormalizeData(df_pred))

    conf_mat = multilabel_confusion_matrix(accu, y_test)
    print('The confusion matrix is\n',conf_mat)

    accuracy = []
    recall = []
    precision=[]
    f1=[]
    for i in output_arr:
        accuracy.append(accuracy_score(accu[i], y_test[i]))
        recall.append(recall_score(accu[i], y_test[i]))
        precision.append(precision_score(accu[i], y_test[i]))
        f1.append(f1_score(accu[i], y_test[i]))
    
    accuracy = np.sum(accuracy)/9
    recall = np.sum(recall)/9
    precision = np.sum(precision)/9
    f1 = np.sum(f1)/9

    print('The average accuracy score is',accuracy)
    print('The average recall score is',recall)
    print('The average precision score is',precision)
    print('The average f1 score is',f1)
    
    print('------------------------------------------------------------------------------------------')
    print('Running KNN')
    
    knn_reg = make_pipeline(KNeighborsRegressor(n_neighbors=3))
    y_pred_knn = []
    for i in output_arr:
        knn_reg.fit(X_train,y_train[i])
        y_pred_knn.append(np.round(knn_reg.predict(X_test),0))
    y_pred_knn
    
    df_pred = pd.DataFrame(np.array(y_pred_knn).T, columns=output_arr)
    
    #predicting optimal values
    opti = df_pred.apply(lambda row: row[row == 1].index, axis=1)
    print(opti)
    
    #Calculating accuracy
    accu = np.round(NormalizeData(df_pred))

    conf_mat = multilabel_confusion_matrix(accu, y_test)
    print('The confusion matrix is\n',conf_mat)

    accuracy = []
    recall = []
    precision=[]
    f1=[]
    for i in output_arr:
        accuracy.append(accuracy_score(accu[i], y_test[i]))
        recall.append(recall_score(accu[i], y_test[i]))
        precision.append(precision_score(accu[i], y_test[i]))
        f1.append(f1_score(accu[i], y_test[i]))
    
    accuracy = np.sum(accuracy)/9
    recall = np.sum(recall)/9
    precision = np.sum(precision)/9
    f1 = np.sum(f1)/9

    print('The average accuracy score is',accuracy)
    print('The average recall score is',recall)
    print('The average precision score is',precision)
    print('The average f1 score is',f1)
    
    print('------------------------------------------------------------------------------------------')
    print('Running MLP')
    
    #Running MLP
    mlp_reg = make_pipeline(MLPRegressor(hidden_layer_sizes=(200,150,100,100,50), max_iter=200, tol=0.5))
    y_pred_mlp = []
    for i in output_arr:
        mlp_reg.fit(X_train,y_train[i])
        y_pred_mlp.append(np.round(mlp_reg.predict(X_test),0))
        
    df_pred = pd.DataFrame(np.array(y_pred_mlp).T, columns=output_arr)
    
    #predicting optimal values
    opti = df_pred.apply(lambda row: row[row == 1].index, axis=1)
    print(opti)
    
    #Calculating accuracy
    accu = np.round(NormalizeData(df_pred))

    conf_mat = multilabel_confusion_matrix(accu, y_test)
    print('The confusion matrix is\n',conf_mat)

    accuracy = []
    recall = []
    precision=[]
    f1=[]
    for i in output_arr:
        accuracy.append(accuracy_score(accu[i], y_test[i]))
        recall.append(recall_score(accu[i], y_test[i]))
        precision.append(precision_score(accu[i], y_test[i]))
        f1.append(f1_score(accu[i], y_test[i]))
    
    accuracy = np.sum(accuracy)/9
    recall = np.sum(recall)/9
    precision = np.sum(precision)/9
    f1 = np.sum(f1)/9

    print('The average accuracy score is',accuracy)
    print('The average recall score is',recall)
    print('The average precision score is',precision)
    print('The average f1 score is',f1)
        
    print('------------------------------------------------------------------------------------------')

def final():
    data = np.loadtxt('tictac_final.txt')
    df = pd.DataFrame(data, columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','y'])
    print('The tic tac toe final boards dataset is\n',df)
    
    #Label encoding
    le = preprocessing.LabelEncoder()

    #Transforming df
    col = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','y']
    for i in range(len(col)):
        df[col[i]]=le.fit_transform(df[col[i]])
        
    #X and Y
    X = df.drop('y', axis = 1)
    y = df['y']
    
    print('The Unique values of y are',y.unique())
    print('----------------------------------------------------------------------------')
    
    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)
    
    print('Running Linear SVM')
    lin_svm = make_pipeline(LinearSVC(random_state=0, tol=1e-5))
    lin_svm.fit(X_train, y_train)
    print('The linear SVM coefficients are',lin_svm.named_steps['linearsvc'].coef_)
    print('The linear SVM intercepts are',lin_svm.named_steps['linearsvc'].intercept_)
    
    y_pred_svm=lin_svm.predict(X_test)
    
    print('Accuracy Score for Linear SVM',accuracy_score(y_test,y_pred_svm))
    print('Recall Score for Linear SVM',recall_score(y_test,y_pred_svm,average='macro'))
    print('Precision Score for Linear SVM',precision_score(y_test,y_pred_svm,average='macro'))
    print('F1 Score for Linear SVM',f1_score(y_test,y_pred_svm,average='macro'))
    
    conf_mat_svm = confusion_matrix(y_test, y_pred_svm)
    print('Confusion matrix for Linear SVm\n',conf_mat_svm)
    
    accuracies_svm = cross_val_score(estimator = lin_svm, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for Linear SVM: {:.2f} %".format(accuracies_svm.mean()*100))
    print("Cross validaton Standard Deviation score for Linear SVM: {:.2f} %".format(accuracies_svm.std()*100))
    print('------------------------------------------------------------------------------------------')
    
    print('Running KNN with 5')
    
    ng = make_pipeline(KNeighborsClassifier(n_neighbors=5))
    ng.fit(X_train, y_train)

    y_pred_ng=ng.predict(X_test)
    
    print('Accuracy Score for KNN',accuracy_score(y_test,y_pred_ng))
    print('Recall Score for KNN',recall_score(y_test,y_pred_ng,average='macro'))
    print('Precision Score for KNN',precision_score(y_test,y_pred_ng,average='macro'))
    print('F1 Score for KNN',f1_score(y_test,y_pred_ng,average='macro'))
    
    conf_mat_ng = confusion_matrix(y_test, y_pred_ng)
    print('Confusion matrix for KNN\n',conf_mat_ng)

    accuracies_ng = cross_val_score(estimator = ng, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for KNN: {:.2f} %".format(accuracies_ng.mean()*100))
    print("Cross validaton Standard Deviation score for KNN: {:.2f} %".format(accuracies_ng.std()*100))
    print('------------------------------------------------------------------------------------------')
    
    print('Running MLP')
    
    mlp = make_pipeline(MLPClassifier(hidden_layer_sizes=(200,150,100,100,50), max_iter=200, tol=0.5))
    mlp.fit(X_train, y_train)

    y_pred_mlp = mlp.predict(X_test)
    
    print('Accuracy Score for MLP',accuracy_score(y_test,y_pred_mlp))
    print('Recall Score for MLP',recall_score(y_test,y_pred_mlp,average='macro'))
    print('Precision Score for MLP',precision_score(y_test,y_pred_mlp,average='macro'))
    print('F1 Score for MLP',f1_score(y_test,y_pred_mlp,average='macro'))
    
    conf_mat_ng = confusion_matrix(y_test, y_pred_ng)
    print('Confusion matrix for MLP\n',conf_mat_ng)
    
    accuracies_mlp = cross_val_score(estimator = mlp, X = X_train, y = y_train, cv = 10)
    print("Cross validation Accuracy score for MLP: {:.2f} %".format(accuracies_mlp.mean()*100))
    print("Cross validaton Standard Deviation score for MLP: {:.2f} %".format(accuracies_mlp.std()*100))
    print('------------------------------------------------------------------------------------------')

def main(var):
    if var == 'single':
        single()
    elif var == 'multi':
        multi()
    elif var == 'final':
        final()
    else:
        print('Invalid keyword')

parser = argparse.ArgumentParser(description='Type')
parser.add_argument('--dataset', help='Input dataset', default='single')
args = parser.parse_args()

if __name__ == '__main__':
    main(args.dataset)


