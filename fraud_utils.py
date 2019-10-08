import pandas as pd
from datetime import datetime
from random import seed, sample
import numpy as np

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

def summary_stats(df):
    df_desc=pd.DataFrame([],index=df.columns)
    df_desc['Unique']=df.nunique()
    df_desc['# Missing']=df.isna().sum()
    df_desc['# Zeros']=[sum(df[col_nm].isin([0,'0'])) for col_nm in df.columns]

    df_desc['Available']=(len(df)-df_desc['# Missing']-df_desc['# Zeros'])
    df_desc['% Available']=100*((df_desc['Available']/len(df)).round(2))
    df_desc['Types']=df.dtypes

    from random import choice, sample
    df_desc['Sample']=[sample(set(df[col_nm].fillna('')),min(10,len(set(df[col_nm])))) for col_nm in df.columns]


    pd.set_option('display.max_rows', 500)
    return df_desc

def build_clean_df(df_dat,df_lbl):
    df_lbl['Fraud']=True
    df_mrg=pd.merge(df_dat,df_lbl,how='left', on='eventId')

    df_mrg.loc[df_mrg.merchantZip.isin(['....','...','0']),'merchantZip']='MISSING'
    df_mrg.loc[df_mrg.merchantZip.isna(),'merchantZip']='MISSING'

    df_mrg.loc[df_mrg.posEntryMode.isin([0]),'posEntryMode']='MISSING'

    df_mrg['reportedTime']=df_mrg['reportedTime'].fillna('')
    df_mrg['TransactionDatetime']=df_mrg['transactionTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    df_mrg['ReportedDatetime']=df_mrg['reportedTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ') if x!='' else '')

    df_mrg['TransactionTimeOfDay']=df_mrg['TransactionDatetime'].apply(lambda x: x.time())
    return df_mrg

def add_prev_transaction(df, fields_to_shift, shift_by=1):
    '''
    Purpose: add features from previous transaction to current transaction
    Inputs: df - transaction level data frame
    Outputs: same dataframe with 'features_to_shift' added after moving them the required number of steps

    Steps:
        - Order by account & transaction time
        - Shift forward by one
        - All cases where accounts no longer matching are to be removed
    Why is this computationally feasible? 
        A: This can be done offline storing the previous transaction 
        on an ongoing basis so it does not impact compute time
    NB: new dataframe has new indexing
    '''
    df_sort=df.sort_values(['accountNumber','TransactionDatetime',]).reset_index(drop=True)
    # Making sure 'accountNumber' is in Fields to shift
    fields_to_shift=list(set(fields_to_shift+['accountNumber','eventId']))
    df_shft=df_sort[fields_to_shift].shift(shift_by).fillna('')
    col_rename={col:'PREV'+col for col in df_shft.columns}
    df_shft=df_shft.rename(columns=col_rename)
    df_new=pd.merge(df,df_shft, how='left',left_on='eventId', right_on='PREVeventId')
    to_clean=list(df_new.accountNumber!=df_new.PREVaccountNumber)
    df_new.loc[to_clean,df_shft.columns]=''

    df_new['DaysSincePrev']=df_new.apply(lambda x: (x['TransactionDatetime']-x['PREVTransactionDatetime']).seconds/3600/24\
                    if x['PREVTransactionDatetime']!='' else 365,axis=1)
    
    df_new=df_new.drop(['PREVaccountNumber','PREVTransactionDatetime','PREVeventId'], axis=1)
    return df_new

def analyze_feature(df,feature):
    df_anz=df[[feature,'eventId','Fraud']].groupby(feature).count()
    df_anz['Fraud%']=df_anz.apply(lambda x: round(100*x['Fraud']/x['eventId'],2),axis=1)
    
    return df_anz.sort_values('eventId', ascending=False)

def account_features(df):
    '''
    Purpose: Feature Engineering: Features based on Account data
    Inputs: df - transaction level data frame
    Outputs: same dataframe with following features added
        - transactionInDomicile - boolean if the transaction is in the same country where most of the transactions have taken place
        - relativeToAcctTransactions - ratio of current transaction to median size of account transactions
        - relativeToAvailableCash - ratio of current transaction to median size of available cash in the account
    '''
    acct_idx=list(set(df.accountNumber))
    acct_idx.sort()
    acct_df=pd.DataFrame([],index=acct_idx)
    # Create account 'Domicile' based on location where most of the transactions have taken place
    # Notion is that fraud may have occured while outside of domiciled country
    acct_ctry_pvt=df.pivot_table('eventId','accountNumber','merchantCountry','count')
    acct_dom=acct_ctry_pvt.idxmax(axis=1)
    # Typical/Median transaction amount
    acct_transact_amt=df[['accountNumber','transactionAmount']].groupby('accountNumber').median()['transactionAmount']
    # Typical/Median level of available Cash
    acct_cash_available=df[['accountNumber','availableCash']].groupby('accountNumber').median()['availableCash']
    
    acct_df['ACCTdomicile']=acct_dom
    acct_df['ACCTtransactionAmount']=acct_transact_amt
    acct_df['ACCTavailableCash']=acct_cash_available
    acct_df=acct_df.reset_index()
    out=pd.merge(df,acct_df,how='left',left_on='accountNumber',right_on='index')
    out['transactionInDomicile']=out.ACCTdomicile==out.merchantCountry
    out['relativeToAcctTransactions']=out.transactionAmount/out.ACCTtransactionAmount
    out['relativeToAvailableCash']=out.transactionAmount/out.ACCTavailableCash
    return out.drop(['index','ACCTdomicile','ACCTtransactionAmount','ACCTavailableCash'], axis=1)


def feat_cat_freq(df,fields,feat_freq):
    '''
    Purpose: generate one hot embeddings based on a maximum number of categories desired, selecting only the most common ones
    Notion that not all features will be relevant. Especially low frequency ones. 
    Allows for a cutoff based on 
     1. proportion of data to be represented (feat_freq < 1 value) ie keep 
     creating one hot encoders until a certain % of data is captured. 
     OR
     2. Number of reatures to be included
    '''
    # 
    # if feat_freq < 1 value is considered to be a percentage of all
    out = df.copy()
    fields = fields if type(fields) == list else list(fields)
    for sel_fld in fields:
        # select categories to convert
        if feat_freq < 1:
            coverage = int(feat_freq*len(out))
            nr_cats = int(min(sum(~(out[sel_fld].value_counts().cumsum()>coverage))+1,len(out)))
            select_cats = list(out[sel_fld].value_counts().index[:nr_cats])
        else:
            nr_cats = feat_freq
            select_cats = list(out[sel_fld].value_counts().index[:nr_cats])
        other_cats = list(set(out[sel_fld].values)-set(select_cats))
        
        out[sel_fld]=out[sel_fld].apply(lambda x: x if x in select_cats else 'other')
        
        df_one_hot = pd.get_dummies(out[[sel_fld]])
        save_params = {'select_cats':select_cats,'other_cats': other_cats}
        control_vars = {'perc_select': sum(out[sel_fld].isin(select_cats)) / len(out[sel_fld]),
                        'tail_length': len(other_cats)}
        
        out=pd.concat([out,df_one_hot],axis=1)
        out=out.drop(sel_fld,axis=1)
    
        print('Field: ' + sel_fld + ': Categorical Selection - ' \
             ' Cover: ' + str(round(control_vars['perc_select'],3)) + \
             ', # selected: ' + str(len(select_cats)) + \
             ', # other: ' + str(len(other_cats)))
    return out

def recall_at_budget(Y_true,X_pred_prob,budget):
    arr_to_sort=X_pred_prob.copy()
    arr_to_sort.sort()
    threshold=arr_to_sort[-int(budget)-1]
    X_pred_at_budget=np.where(X_pred_prob>threshold,1,0)
    return recall_score(np.array(Y_true).reshape((len(Y_true))),X_pred_at_budget,)

def pipe_initial(df_mrg, tight=True):
    '''
    Pipeline 1: Converts merged transaction and label data into
    
    
    '''
    if tight:
        params=[.99,.8,40]
    else:
        params=[.99,.95,.6]
    df_out=df_mrg.pipe(feat_cat_freq,['merchantCountry','posEntryMode'],params[0])\
                 .pipe(feat_cat_freq,['mcc'],params[1])\
                 .pipe(feat_cat_freq,['merchantId','merchantZip'],params[2])

    df_out['LOGtransactionAmount']=df_out.transactionAmount.clip(lower=1).apply(lambda x: np.log(x))
    df_out['transactionToTotalAvailable']=df_out.apply(lambda x: \
                                                       x['transactionAmount']/x['availableCash'], axis=1).clip(0,.2)
    X=df_out.drop(['transactionTime', 'eventId', 'accountNumber', 'transactionAmount',
       'availableCash', 'reportedTime', 'Fraud','TransactionDatetime', 'ReportedDatetime','TransactionTimeOfDay'], axis=1)


    Y=df_out['Fraud'].fillna(False)
    return X,Y

def pipe_w_prev(df_mrg, tight=True):
    '''
    Pipeline 2: Converts df_mrg (cleaned up original data) to X, Y frames ready for model ingestion
    
    Generates the following features:
    - Account features
    - Features from previous transaction (can one-hot encodes them)
    - One hot encoding for all other data features
    
    Has options tight=True/False which include different number of features from 
    two sets of predefined feature fraction parameters in order to keep computation fairly efficient
    See notes from account_features & add_prev_transaction for more
    
    
    '''
    
    #predefined feature fraction parameters
    if tight:
        params=[.99,.8,40]
    else:
        params=[.99,.95,100]
    
    df_out3=df_mrg.pipe(account_features)\
             .pipe(add_prev_transaction , ['posEntryMode','TransactionDatetime','mcc','merchantCountry'],\
                        shift_by=1)\
             .pipe(feat_cat_freq,['merchantCountry','posEntryMode','PREVmerchantCountry','PREVposEntryMode'],params[0])\
             .pipe(feat_cat_freq,['mcc','PREVmcc'],params[1])\
             .pipe(feat_cat_freq,['merchantId','merchantZip'],params[2])
                
    X=df_out3.drop(['transactionTime', 'eventId', 'accountNumber', 'transactionAmount', \
                    'availableCash', 'reportedTime', 'Fraud','TransactionDatetime', \
                    'ReportedDatetime','TransactionTimeOfDay'], axis=1)


    Y=df_out3['Fraud'].fillna(False)
    return X,Y

def pipe_w_prev_wide(df_mrg):
    '''
    Not tight version of Pipe 2
    '''
    return pipe_w_prev(df_mrg, tight=False)


def big_loop(df_mrg,
             data_pipes, 
             data_pipe_nm,
             budget,
             n_estimators=1000,
             downsample_folds=5,
             label_ratios=3, 
             verbose=0):
    
    '''
    Purpose: Run through all selected models and different splits of the data
        - Takes a number of different pipelines that would be run through the models
        - Label ratios allow to select the proportions of negative to positive labels to consider
        - Downsample folds will iterate through different random selections of negative labels
        - Uses downsampled data to train the models but assesses results on the full test data
        - Measure is the recall (% accurately identified fraud) under budget constrain
    Inputs:
        df_mrg - merged clean frame from input data (not fully numeric, yet)
        n_estimators - nr_estimator parameters for decision tree models
        label_ratios - Nr of different negative to positive label ratios to consider. 2 means will use 1:1 and 2:1
    Outputs:
        Summary table of experiment results
    '''
    seed(1)
    ls_exp_res=[]
    for (iii,pipe) in enumerate(data_pipes):
        X,Y=pipe(df_mrg)
        print(iii,'Pipe done') if verbose>0 else ''
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=.25, random_state=42)

        for lbl_ratio in range(1,label_ratios+1):
            all_folds_idx=sample(list(Y_train[Y_train!=True].index),lbl_ratio*downsample_folds*sum(Y_train))

            for i in range(downsample_folds):
                down_idx=all_folds_idx[lbl_ratio*i*sum(Y_train):lbl_ratio*(i+1)*sum(Y_train)]+\
                                    list(Y_train[Y_train==True].index)
                down_idx.sort()
                Y_down=Y_train.loc[down_idx]
                X_down=X_train.loc[down_idx]
                
                sel_ = SelectFromModel(XGBClassifier(n_estimators=n_estimators))
                sel_.fit(X_down.fillna(0), Y_down)
                selected_feat_xgb = X_down.columns[(sel_.get_support())]
                print(selected_feat_xgb) if verbose>0 else ''

                sel_ = SelectFromModel(LGBMClassifier(n_estimators=n_estimators))
                sel_.fit(X_down.fillna(0), Y_down)
                selected_feat_lgbm = X_down.columns[(sel_.get_support())]
                print(selected_feat_lgbm) if verbose>0 else ''
                
                print(iii,lbl_ratio,'Data slices done', datetime.now()) if verbose>0 else ''

                gbm = GradientBoostingClassifier(n_estimators=n_estimators)
                ls_exp_res=run_classifier(X_down,
                                  Y_down,
                                  X_test,
                                  Y_test,
                                  gbm, 
                                  'GBM',
                                  ls_exp_res,budget,
                                  lbl_ratio,
                                  data_pipe_nm[iii],i)
                print(iii,lbl_ratio,'GBM', datetime.now())   if verbose>0 else ''

                xgboost = XGBClassifier(n_estimators=n_estimators)
                ls_exp_res=run_classifier(X_down,
                                  Y_down,
                                  X_test,
                                  Y_test,
                                  xgboost, 
                                  'XGB',
                                  ls_exp_res,budget,
                                  lbl_ratio,
                                  data_pipe_nm[iii],i)
                print(iii,lbl_ratio,'XGB', datetime.now()) if verbose>0 else ''

                lightgbm = LGBMClassifier(n_estimators=n_estimators)
                ls_exp_res=run_classifier(X_down,
                                          np.array(Y_down).reshape((len(Y_down))),
                                          X_test,
                                          Y_test,
                                          lightgbm, 
                                          'LGBM',
                                          ls_exp_res,budget,
                                          lbl_ratio,
                                          data_pipe_nm[iii],i)
                print(iii,lbl_ratio,'LGBM', datetime.now())   if verbose>0 else ''

                xgboost =XGBClassifier(n_estimators=n_estimators)  
                ls_exp_res=run_classifier(X_down[selected_feat_xgb],
                                          Y_down,
                                          X_test[selected_feat_xgb],
                                          Y_test,
                                          xgboost, 
                                          'XGB-Select',
                                          ls_exp_res,budget,
                                          lbl_ratio,
                                          data_pipe_nm[iii],i)
                print(iii,lbl_ratio,'XGB-Select', datetime.now())  if verbose>0 else ''


                lightgbm = LGBMClassifier(n_estimators=n_estimators)
                ls_exp_res=run_classifier(X_down[selected_feat_lgbm],
                                          np.array(Y_down).reshape((len(Y_down))),
                                          X_test[selected_feat_lgbm],
                                          Y_test,
                                          lightgbm, 
                                          'LGBM-Select',
                                          ls_exp_res,budget,
                                          lbl_ratio,
                                          data_pipe_nm[iii],i)
                print(iii,lbl_ratio,'LGBM-Select', datetime.now())      if verbose>0 else ''


    df_exp_res=pd.DataFrame(ls_exp_res,columns=['ExperimentID','Label Ratio','Model','Pipe','Budget','Result'])
    return df_exp_res

def run_classifier(X_down,Y_down,
                    X_test,Y_test,
                    classif, classif_nm,
                    ls_exp_res, budget,
                    lbl_ratio,data_pipe_nm,i):
    '''
    Helper function for the big loop to initiate, fit run and store results from each model run
    '''
    classif.fit(X_down, Y_down)
    X_pred_prob=classif.predict_proba(X_test)[:,1]
    res1=recall_at_budget(Y_test,X_pred_prob,budget)
    res2=recall_at_budget(Y_test,X_pred_prob,budget*1.5)
    res3=recall_at_budget(Y_test,X_pred_prob,budget*2)
    ls_exp_res=ls_exp_res+[[i,str(lbl_ratio)+str(':1'),classif_nm,data_pipe_nm,'1x budget',res1]]
    ls_exp_res=ls_exp_res+[[i,str(lbl_ratio)+str(':1'),classif_nm,data_pipe_nm,'1.5x budget',res2]]
    ls_exp_res=ls_exp_res+[[i,str(lbl_ratio)+str(':1'),classif_nm,data_pipe_nm,'2x budget',res3]] 
    return ls_exp_res


def plot_cv_results(X,Y,parameters,classifier):  
    key_list=[list(par.keys())[0] for par in parameters]
    fig, axs = plt.subplots(int(np.ceil(len(key_list)/4)),4, 
                            figsize=(20, int(3*np.ceil(len(key_list)/4))), 
                            facecolor='w', edgecolor='k')

    axs = axs.ravel()

    for i , key in enumerate(key_list):
        dt0=datetime.now()
        clf = GridSearchCV(classifier, parameters[i], cv=5, n_jobs=4, scoring='precision')
        clf.fit(X,Y,)
        print('Precision', clf.best_params_, 'Best score', clf.best_score_, 'Time',str(datetime.now()-dt0)) 
        clf_res=frame_cv_results(clf)
        axs[i].plot([str(tick) for tick in parameters[i][key]],clf_res['mean_test_score']\
                    ,label='Precision')
        
        dt0=datetime.now()
        clf = GridSearchCV(classifier, parameters[i], cv=5, n_jobs=4, scoring='recall')
        clf.fit(X,Y,)
        print('Recall', clf.best_params_, 'Best score', round(clf.best_score_,3), 'Time',str(datetime.now()-dt0)) 
        clf_res=frame_cv_results(clf)
        axs[i].plot([str(tick) for tick in parameters[i][key]],clf_res['mean_test_score']\
                    ,label='Recall')
        
        axs[i].set_title(key) 
        #axs[i].set_xticklabels([str(tick) for tick in parameters[i][key]])   
        if i == len(key_list)-1:
            axs[i].legend()
    plt.setp(axs, yticks=np.linspace(0,1,6))

def frame_cv_results(clf):
    df_cv=pd.DataFrame()

    for i in range(len(clf.cv_results_['params'])):
        df_cv=df_cv.append(clf.cv_results_['params'][i],ignore_index=True)

    metrics=['mean_test_score', 'std_test_score', ]

    for metr in metrics:
        df_cv[metr]=clf.cv_results_[metr]
    return df_cv