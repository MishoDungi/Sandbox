import random
from sklearn.metrics import precision_score, recall_score
# NN Keras Modeling Requirements
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.models import Model
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.95, loss="mse", optimizer=Adam(lr=0.01)):

    model = Sequential()

    nn = 'deep'#'deep'
    if nn == 'deep':
        model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(LSTM(neurons, return_sequences=True, dropout=0.5))
        model.add(LSTM(neurons, return_sequences=True, dropout=0.5))
        model.add(LSTM(neurons, return_sequences=True, dropout=0.5))
        model.add(LSTM(neurons, return_sequences=True, dropout=0.5))
        model.add(LSTM(neurons, return_sequences=True, dropout=0.5))
        model.add(LSTM(neurons, return_sequences=True, dropout=0.5))
        model.add(LSTM(int(neurons), dropout=0.5))
    elif nn == 'deep1':
        model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(dropout))
        model.add(LSTM(neurons, input_shape=inputs.shape[2]))
    elif nn == 'rnn':
        #model.add(Bidirectional(LSTM(neurons, return_sequences=True), input_shape=(inputs.shape[1], inputs.shape[2]) ))
        #model.add(GlobalMaxPool1D())
        model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    
    return model

def gen_model_data(data, window, intervals, cat_labels,cats=True):
    # Wrap a time series including 'Adj Close', 'Volume' & 'Range'
    # Prepare it for an RNN where we use 'window'=10 days of lag to predict the next day Y

    cols=list(data.columns[:-2])

    X = []
    #print(len(data))
    for i in range(len(data)-window):
        temp_set = np.array(data[cols][i:i+window].copy())
        X.append(temp_set)
        
    #print('.')
    X = [np.array(X) for X in X]
    X = np.array(X)

    Y=data['Y'][window:len(data)]
    if cats == True:
        Y=pd.get_dummies(pd.cut(Y,bins=intervals,labels=cat_labels))
    return X,Y

# NB assumes 3 categories
def gen_downsmpl_data(x,y):
    # decide how many large moves we have observed
    avg_lrg_moves=y.sum()[[0,2]].mean()
    # Number to drop is total nr cat 1 minus small categories
    nr_to_drop=int(y[1].sum()-avg_lrg_moves)
    # index of all members of category 1 (i.e. neutral/small moves category)
    idx_sml_moves=list(y[y[1]==1].index)
    # choose at random which to drop    
    lst_to_drop=random.sample(idx_sml_moves,nr_to_drop)
    smpl_list=list(set(y.index)-set(lst_to_drop))
    Y_smpl=y.loc[smpl_list]
    X_smpl=x[smpl_list]
    assert(len(Y_smpl)==len(X_smpl))
    print('New sample total size ' + str(len(Y_smpl)))
    return X_smpl, Y_smpl

def gen_feats(stk, stk_nr):
    stk['range']=(stk['Adj. High']-stk['Adj. Low'])/stk['Adj. Close']
    stk['ret_cls']=(stk['Adj. Close'].shift(-1)-stk['Adj. Close'])/stk['Adj. Close']
    stk['ret_opn']=(stk['Adj. Open'].shift(-1)-stk['Adj. Open'])/stk['Adj. Open']
    stk['ret_vol']=(stk['Adj. Volume'].shift(-1)-stk['Adj. Volume'])/stk['Adj. Volume']
    stk['Y']=stk['ret_cls'].shift(-1)

    stk_feat=stk[['range','ret_cls','ret_opn','ret_vol','Y']].copy()[:-2]
    stk_feat=stk_feat.replace([-np.inf,np.inf,np.nan],[-1,1,0])
    
    scaler = preprocessing.StandardScaler()
    stk_scld = pd.DataFrame(scaler.fit_transform(stk_feat),columns=stk_feat.columns)
    stk_feat['stk_id']=stk_nr
    return stk_feat, stk_scld

class_names = ['-','0','+']
intervals=[-1000,-0.02,0.02,1000]
cat_labels=[0,1,2]

def confusion_calc(X,Y,model,intervals=intervals,cat_labels=cat_labels,thresholds=False,cats=True):
    if cats == True:
        y_pred  = model.predict(X)
        if thresholds == True:
            y_thresh=np.zeros(y_pred.shape)
            thresh=np.percentile(y_pred, 67, axis=0, keepdims=True)
            y_thresh=y_pred>[thresh[0]]*y_pred.shape[0]
            y_pred=y_thresh*1
        y_pred_cat = y_pred.argmax(axis=1)
        y_cat=np.array(Y).argmax(axis=1)
    else:
        y_pred = model.predict(X)
        y_pred_cat =pd.cut(pd.Series(y_pred.flatten()),bins=intervals,labels=cat_labels)
        y_cat=pd.cut(y_train,bins=intervals,labels=cat_labels)

    return y_cat, y_pred_cat

def cm_analysis(y, y_pred, class_names, plot=True):
    if plot == True:
        np.set_printoptions(precision=2)
        cnf_matrix = confusion_matrix(y, y_pred)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        plt.show()
    '''
    for i in range(3):
        if i==0 or i==2:
            print('Class: ' + class_names[i] + ' precision: '+
                  str(np.round(precision_score(y==i,y_pred==i),4)))
            #print(recall_score(y==i,y_pred==i))
    '''
    return [np.round(precision_score(y==0,y_pred==0),4),np.round(precision_score(y==2,y_pred==2),4)]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        1
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

