
from fraud_utils import *
import pandas as pd
from sklearn.model_selection import train_test_split



def run_basic_stats(df_lbl,df_dat):
    print("% fraud in population" ,len(df_lbl)/len(df_dat))
    print(summary_stats(df_dat))





def run_all(run_stats=False,
        fraud_by_feature=False,
        run_big_loop=False,
        run_hyperparameters=False):

    '''
    Use this file to run different parts of the analysis
      1. Setting run_stats to True will run some statistics on the data
      2. Setting fraud_by_feature to True runs a feature by feature summary of % fraud
      3. Setting run_big_loop to True will run the entire loop of model experiments
      4. Setting run_hyperparameters to True will run hyperparameter analysis
    '''
    # Load files from local directory and merge and clean them into one frame
    df_dat=pd.read_csv('transactions_obf.csv')
    df_lbl=pd.read_csv('labels_obf.csv')
    df_mrg=build_clean_df(df_dat,df_lbl)

    if run_stats:
        run_basic_stats(df_lbl,df_dat)

    if fraud_by_feature:
        flds=['merchantId','merchantCountry','transactionAmount','mcc','posEntryMode',\
            'accountNumber','TransactionTimeOfDay','merchantZip']
        

        # Multiple of the overall average 0.7% to show
        threshold_multiple=3
        for (i,fld) in enumerate(flds):
            res_df=analyze_feature(df_mrg,fld)
            outlier_df=res_df[(res_df.eventId>.01*len(df_mrg))&(res_df['Fraud%']>threshold_multiple*.7)]
            if len(outlier_df)>0:
                print(outlier_df)

    if run_big_loop:
        # Setting parameters for the big loop
        data_pipes=[pipe_initial,pipe_w_prev,pipe_w_prev_wide]
        data_pipe_nm=['Pipeline 1','Pipeline 2', 'Pipeline 3']
        downsample_folds=1 # Use 3 for results validation

        budget=3*400
        n_estimators=5 # Use 1000 for results validation

        # Big loop
        df_results=big_loop(df_mrg,
                    data_pipes, 
                    data_pipe_nm,
                    budget,
                    n_estimators=n_estimators,
                    downsample_folds=downsample_folds,
                    label_ratios=3, 
                    verbose=1)

        #prettyfy-ing results
        df_results2=df_results.copy()
        df_results2['Result']=df_results.apply(lambda x: round(x['Result'],3),axis=1)

        piv=df_results2.pivot_table('Result',['Budget','Model'],['Pipe','Label Ratio',]).loc[['1x budget','1.5x budget','2x budget']]
        print(piv)

    if run_hyperparameters:
        parameters_GBM = [
            {"learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2]}, #
            {"max_depth":[1,5,20,30]},
            {"subsample":[0.25,0.5, 0.75, 1.0]},
            {"n_estimators":[100,500,1000]}
        ]


        parameters_LGBM = [
            {"reg_alpha":[.0001,.001,0.01,.1]}, 
            {"reg_lambda":[0.0,.5,1]},
            {"num_leaves":[5,15,31,50]},
            {"learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2]}, 
            {"max_depth":[1,5,20,30]},
            {"subsample":[0.25,0.5, 0.75, 1.0]},
            {"n_estimators":[100,500,1000]}
        ]

        parameters_XGB = [
            {"reg_alpha":[.0001,.001,0.01,.1]}, 
            {'max_depth':[1,5,20,30]},
            {"gamma":[0.25,0.5, 0.75, 1.0]},
            {"subsample":[0.25,0.5, 0.75, 1.0]},
            {"colsample_bytree":[0.25,0.5, 0.75, 1.0]},
            {"n_estimators":[100,500,1000]},
            {"learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2]}, 
        ]

        X,Y=pipe_w_prev(df_mrg)

        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=.25, random_state=42)

        all_folds_idx=sample(list(Y_train[Y_train!=True].index),sum(Y_train))

        down_idx=all_folds_idx[:sum(Y_train)]+list(Y_train[Y_train==True].index)
        down_idx.sort()
        Y_down=Y_train.loc[down_idx]
        X_down=X_train.loc[down_idx]

        plot_cv_results(X_down,Y_down,parameters_LGBM,LGBMClassifier())
        plot_cv_results(X_down,Y_down,parameters_XGB,XGBClassifier())
        plot_cv_results(X_down,Y_down,parameters_GBM,GradientBoostingClassifier())


if __name__ == "__main__":

    
    print('Use this file to run different parts of the analysis')
    print('1. Setting run_stats to True will run some statistics on the data')
    print('2. Setting fraud_by_feature to True runs a feature by feature summary of % fraud')
    print('3. Setting run_big_loop to True will run the entire loop of model experiments')
    print('4. Setting run_hyperparameters to True will run hyperparameter analysis')

    run_stats=True
    fraud_by_feature=True
    run_big_loop=False
    run_hyperparameters=False
    run_all(run_stats=run_stats,
        fraud_by_feature=fraud_by_feature,
        run_big_loop=run_big_loop,
        run_hyperparameters=run_hyperparameters)