# ML Analytical Functions

import os
import pickle
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from workflow_config import *


from sklearn.model_selection import cross_val_score, KFold,cross_val_predict,cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,\
classification_report,confusion_matrix,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier


# for metrics:
# https://scikit-learn.org/stable/modules/model_evaluation.html


def v_vs_I_Plot(lst_plots,save_file_path,save_file_name,save_file_flag,title=''):
    for indx in lst_plots:
        plt.scatter(indx[1],indx[0],label=indx[2])
        plt.xlabel('V', fontsize = 16)
        plt.ylabel('I', fontsize = 16)
        plt.yticks(fontsize = 13)
        plt.xticks(fontsize = 13)
        plt.gca().yaxis.offsetText.set_fontsize(16)
        plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        plt.legend(loc='best')
        plt.tight_layout()
        if save_file_flag:
            plt.savefig(os.path.join(save_file_path,f'I-V_Plot_{save_file_name}_{len(lst_plots)}.pdf'),dpi = 300, bbox_inches='tight')
        else:
            plt.title(title)
            
            
def call_plot(df_data):
    groups = df_data.groupby('Cycle')
    fig, ax = plt.subplots()
    plt.xlabel('V')
    plt.ylabel('I')
    for cycle_id, group in groups:
        ax.scatter(group.Ewe, group.I, label=f'Cycle: {cycle_id}')
    ax.legend(loc='best')
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

#     plt.title(f'I vs V Measurements: file_name:{SP200_result_file_sig}\n'
#               f'Ei:{Ei},E1:{E1},E2:{E2},Ef{Ef},Scan_number:{cv_Scan_number},Record_every_dE:{cv_Record_every_dE}' \
#     f'Average_over_dE:{cv_Average_over_dE}\n,N_Cycles:{cv_N_Cycles},Begin_measuring_I:{cv_Begin_measuring_I}' \
#     f'End_measuring_I:{cv_End_measuring_I},E_range:{cv_E_range},I_range:{cv_I_range},\n' \
#     f'Bandwidth:{cv_Bandwidth},tb:{cv_tb},ScanRate:{ScanRate_lst},vs_initial:{vsinitial_lst}',fontsize=8)
#     plt.savefig(f'{result_file_path}\{SP200_result_file_sig}.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    
###################################################################
###################################################################
def call_GPR_for_probing(_X,_y_target,x_probe,save_file_path,save_file_name,save_file_flag,show_IV_plot):
    
    # GPR
    #1) _X  samples matrix. It represents the voltage potential (Ewe).
    #2) _y target values thqat represents the current (I).
    #3) x_probe: a vector of random potential samples in the range of CV Scan_Rate.
    #4) _X and _y are usually expected to be numpy arrays or equivalent array-like data types,
    # though some estimators work with other formats such as sparse matrices.
    cross_val=cross_val_score
    GPR=GaussianProcessRegressor()

    GPR.fit(_X,_y_target)
    scores = cross_val(GPR, _X,_y_target,cv=5)
    _y_pred = GPR.predict(_X)
    GPR_mse = mean_squared_error(_y_target,_y_pred,squared=True)  

    i_probe=GPR.predict(x_probe)
    
    # print( "V_range: min:  {:.4f} , max: {:.4f}".format(_X.min(0)[0],_X.max(0)[0]))
    # print("Mean cross-validataion score: %.2f" % scores.mean())
    # print("MSE: %f" % GPR_mse)
    # print("RMSE: %f" % np.sqrt(GPR_mse))

    plt.figure(figsize=(10,5))
    #plt.suptitle(f' CV Measurement Analysis: {save_file_name}')
    ax1 = plt.subplot(1, 2, 1)
    v_vs_I_Plot([(_y_target,_X,'Measurments')],save_file_path,save_file_name,save_file_flag,title='I-V Plot')

    #####################################
    ax2=plt.subplot(1, 2, 2)
    v_vs_I_Plot([(_y_pred,_X,'Regression'),(i_probe,x_probe,'Regression Probing')],save_file_path,save_file_name,save_file_flag,title='I-V Regression and Probing')

    print(f'show_IV_plot: {show_IV_plot}')
    if show_IV_plot:
        plt.show()
    else:
        plt.clf() 
    
    return i_probe  


###########################################################
###########################################################
def GPR_for_CV_feature_extraction(training_data_path,v_probe,save_file_flag,show_IV_plot):    
    i_probe_lst=[]
    for root, dirs, files in os.walk(training_data_path):
        file_cnt=0;
        for file in files:
            if file.endswith('.txt'):
                print(file)
                i_probe_rcrd={}
                df = pd.read_csv(os.path.join(root,file),sep='\t')
                a=np.array(df.Ewe).reshape(-1,1)
                b=np.array(df.I).reshape(-1,1)
                i_probe_buff=call_GPR_for_probing(a,b,v_probe,training_data_path,file,save_file_flag,show_IV_plot)
                file_cnt=file_cnt+1;
                i_probe_rcrd['indx']=file
                i_probe_rcrd['data']=i_probe_buff.flatten()
                i_probe_lst.append(i_probe_rcrd)
    return i_probe_lst
        
    
#############################################################
#############################################################
def call_assign_classes(i_probe_lst):
    # Assign Classes to the training data set
    # Classes represent "Valid" and "invalid" tests. 
    # These are based on a signituare included in the file name of how they are collected.
               
            
    for i in range(len(i_probe_lst)):
        if 'GOOD' in i_probe_lst[i]['indx']:
            i_probe_lst[i]['class']=1
        else:
            i_probe_lst[i]['class']=0

    return i_probe_lst

###########################################################
###########################################################
def call_RF(_X,_y_target):    
   
    clf = RandomForestClassifier()
    clf.fit(_X,_y_target)

    #scores = cross_val(clf, _X,_y_target)
    y_pred=clf.predict(_X)
    #print("Mean cross-validataion score: %.2f" % scores.mean())
    cm=confusion_matrix(_y_target,y_pred)
    return clf,cm


    

###########################################################
###########################################################
def call_Train_n_Serialize_RF_Classifier(i_probe_lst,EoT_Classifier_Path, EoT_Classifier):

    # Train a classifer with the probing/extrapolated measurement
    X=np.array([list(i_probe_lst[i]['data']) for i in range(len(i_probe_lst))])
    #y=np.reshape(pd.DataFrame(i_probe_lst)['class'].to_numpy(), (-1, 1))
    
    y=pd.DataFrame(i_probe_lst)['class'].to_numpy()
    clf_trained,conf_mat=call_RF(X,y)
    #print('confusion_matrix\n',conf_mat) 

    # Serialize and store the classifier
    ofile=open(os.path.join(EoT_Classifier_Path, EoT_Classifier),'wb')
    pickle.dump(clf_trained,ofile)
    ofile.close()
    
    
    
###################################################################
###################################################################
def call_ML_measurement_validation(profile,v_probe,EoT_Classifier_Path, EoT_Classifier,save_file_flag,show_IV_plot):
    # De-serialize and load the classifier
    clf_retrieved=pickle.load(open(os.path.join(EoT_Classifier_Path, EoT_Classifier),'rb'))
    df = pd.read_csv(profile,sep='\t')
    a=np.array(df.Ewe).reshape(-1,1)
    b=np.array(df.I).reshape(-1,1)
    #print(profile)
    i_probe_df=call_GPR_for_probing(a,b,v_probe,os.path.dirname(profile),os.path.basename(profile),save_file_flag,show_IV_plot)
    ml_status=clf_retrieved.predict(i_probe_df.reshape(1, -1))
    return ml_status[0]

###################################################################
###################################################################
def call_analyze_CV_profile(New_File,v_probe,testing_data_path,EoT_Classifier_Path, EoT_Classifier,save_file_flag,show_IV_plot):
    target_profile=os.path.join(testing_data_path,New_File)
    status=call_ML_measurement_validation(target_profile,v_probe,EoT_Classifier_Path, EoT_Classifier,save_file_flag,show_IV_plot)
    return target_profile,status

