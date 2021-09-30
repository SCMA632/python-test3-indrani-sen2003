
def hello_world():
    import pandas as pd;
    df=pd.read_csv("Walmart2.csv");
    df.dtypes
    df.head(15)
    
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import statsmodels.api as sm
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import os

#os.chdir("C://Users//91810//Desktop//SCMA")
    nsso=pd.read_csv("4. NSSO68 data set.csv")

    columns = ['Age', 'Religion', 'Sex', 'Social_Group', 'MPCE_MRP', 'emftt_q','state_1']
    nsso_new=nsso[columns]
    MP=nsso_new[nsso_new['state_1']=='MP']
    MP['emftt_q'].value_counts()

    MP.dropna(inplace = True)

    MP.loc[(MP['emftt_q'] > 0.0),'emftt_q'] = 1
    MP.loc[(MP['emftt_q'] == 0.0),'emftt_q'] = 0

    MP['emftt_q'] = MP['emftt_q'].astype(int)
    MP['emftt_q'].value_counts()

    del MP['state_1']

    X = MP.drop(['emftt_q'], axis = 1)
    y = MP['emftt_q']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    X_train.shape, X_test.shape

    X_train = sm.add_constant(X_train)
    X_train.info()

    model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
    model.summary()

    df = pd.DataFrame()
    df['coef'] = model.params
    df['pvalues'] = model.pvalues
    sig_vars = df[df['pvalues'] < 0.05]
    sig_vars['expon'] = np.exp(df['coef'])
    sig_vars

    test_pred = model.predict(sm.add_constant(X_test))
    test_pred

    test_df = pd.DataFrame()
    test_df['y_actual'] = y_test
    test_df['y_prob'] = test_pred
    test_df['y_pred'] = np.where(test_df['y_prob'] < 0.5, 0, 1)
    test_df

    conf_matrix = confusion_matrix(test_df['y_actual'], test_df['y_pred'],[1,0])
    conf_matrix

    TP = conf_matrix[0][0]
    FN = conf_matrix[0][1] 
    FP = conf_matrix[1][0]
    TN = conf_matrix[1][1]

    TPR = TP / (TP + FN)*100
    TNR = TN / (TN + FP)*100
    Precision = TP / (TP+FP)*100
    Accuracy = (TP + TN) / (TP+FN+FP+TN)*100
    FPR = 100-TNR

    print('TPR: ', round(TPR))
    print('TNR: ', round(TNR))
    print('Precision: ', round(Precision))
    print('Accuracy: ', round(Accuracy))
    print('FPR: ', round(FPR))

    sns.heatmap(conf_matrix, annot=True, fmt='.2f',
    xticklabels=['1', '0'],
    yticklabels=['1', '0'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label');

    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(test_df['y_actual'],test_df['y_prob'], drop_intermediate=False)
    auc_score = metrics.roc_auc_score(test_df['y_actual'], test_df['y_prob'])
    plt.figure(figsize = (6, 4))
    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0,1.05)
    plt.xlabel('False Positive Rte or [1 - TNR]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve')
    plt.legend(loc = 'lower right');
    return(1)
