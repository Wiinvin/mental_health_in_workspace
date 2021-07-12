#!/usr/bin/env python
#
# file: $(EXP)/python/src/analyze.py
#
# revision history:
#
# 20210525 (VS): initial version
#
#------------------------------------------------------------------------------

# import system modules
#
import numpy as np
np.random.seed(13)
import random
random.seed(13)
import os
import pandas as pd
from collections import Counter

# import ML modules
#
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    cross_val_score, GridSearchCV

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, recall_score



## import support modules
#
import preprocess_dat as prep
import seaborn as sns
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#
# global variables
#
#------------------------------------------------------------------------------

OUT_SUMMARY_FILE = "summary.txt"
DEF_RAND_STATE = 113
ORDINAL_ENCODING = False
PLOT_F = True

# method: run_analysis
#
# arguments: 
#  csv_file_a: input file as pandas data frame
#  params_a: parameters for the experiment
#  odir_a: output location where analysis files will be saved
#
# return: 
#  status: status if the function executed successfully
#
# This function is input data specific functions which filters data and runs analysis
# using some expert algorithm
#
def run_analysis(csv_file_a, params_a, odir_a):

    ## initialize necessary variables
    #
    status = True
    ignore_cols = ['timestamp','comments']
    ofile = os.path.join(odir_a, OUT_SUMMARY_FILE)

    ## load the parameters
    #
    normalize_f = params_a['normalize_f']  ## is kept false because we want high var in mdl's performance
    normalize_method = params_a['norm_type']
    preproc_feattype = params_a['preproc_feattype'] ## ordinal.. NNs can learn things
    categorical_f = params_a['categorical_f'] ## it's binary classification but for generalization

    ## load data
    #
    raw_dat = pd.read_csv(csv_file_a)
    dat_head = raw_dat.columns

    #################################
    ## generic preprocessing step  ##
    #################################

    ## first do analysis on the unstructured data
    ## that means any time-series related labels should be dropped
    #
    unstructured_dat = prep.Preprocess(raw_dat, params_a, ignore_cols)

    ####################################################
    ## fine tune based on what we know about the data ##
    ####################################################

    ## replace invalid entries with NaN (negative ages, non-defined genders)
    #
    #prep.correct_ages(raw_dat['age'])
    prep.update_ages(raw_dat)

    ## map the string with limited number of classes
    #
    prep.update_genders(raw_dat)

    ## id all the string based entries
    ## here we collect all the tools which will help us quickly separate different
    ## sets of data..
    #
    uniq_entries = dict()
    str_ids = dict()
    numeric_col = ['age']
    lbl_col = ['treatment']
    nan_idx = []
    unstructured_dat.collect_unique_values(raw_dat, uniq_entries)
    unstructured_dat.assign_ids2strs(uniq_entries, str_ids, numeric_col)
    
    ## collect row indices which include NaN values
    ## We ignore all the state and work_interfere values because it has very high
    ## number of missing values. Although, state info could be very important for the 
    ## domestic analysis and work_interference can be used to more accuractely create
    ## labels. We will explore this later and implement better logics
    #
    #print(raw_dat.isna().sum())
    raw_dat.drop(columns=ignore_cols+['country', 'state', 'work_interfere'], inplace = True)
    unstructured_dat.flag_nans(raw_dat, nan_idx)
    unstructured_dat.rm_entries(raw_dat, nan_idx)    

    ## We will not use imputation methods to fill in the Nan values because it is
    ## relatively smaller data with a lot of missing entries/outliers. It could bias
    ## the overall trend.. Instead let's just drop those samples (It's a trade-off)
    ## we also drop columns with higher nan values
    #

    ## basic analysis
    ## This will be independent analysis for age, gender, country, etc. variables
    ## This analysis will help us understand important factor and biases of the data
    ## which in turn, will help us tune the hyperparameters of the more sophisticated
    ## models later..
    #
    target_events = ['age', 'gender', 'no_employees', 'tech_company', 'remote_work', 'family_history']    
    perform_rudimentary_analysis(raw_dat, target_events, uniq_entries, ofile)

    if preproc_feattype == "ordinal":
        ## separate columns which can be used as features and labels
        ## For now, we will ignore work_interfere label which could be a strong
        ## indicator of a mental illness and can be used as a label
        #
        dat_arr, lbl_arr, feat_names = prep.gen_ordinal_feats_labs(raw_dat, str_ids,
                                                           lbl_col, numeric_col)


        


    elif preproc_feattype == "onehot":
        dat_arr, lbl_arr, feat_names = prep.gen_onehot_feats_labs(raw_dat, str_ids,
                                                           lbl_col, numeric_col)

    ## collect categorical labels
    #
    lbl_arr_oned = np.reshape(lbl_arr, (-1,1))


    if categorical_f:
        categorical_lbls = preprocessing.OneHotEncoder().fit_transform(lbl_arr_oned).toarray()

        x_tr, x_test, y_tr, y_test = train_test_split(dat_arr, categorical_lbls,
                                                      stratify = lbl_arr,
                                                      test_size = 0.3,
                                                      random_state = DEF_RAND_STATE)

    else:
        x_tr, x_test, y_tr, y_test = train_test_split(dat_arr, lbl_arr_oned,
                                                      stratify = lbl_arr,
                                                      test_size = 0.3,
                                                      random_state = DEF_RAND_STATE)

    ## normalize if necessary
    #
    if normalize_f:
        ## scale the data
        ## Since outliers have been already removed it is better to scale data with a
        ## min_max scalar
        ## TODO: scale the first dimension (Age) separately
        #
        minmax_abs_scalar = preprocessing.MinMaxScaler()
        minmax_abs_scalar.fit_transform(x_tr)
        x_tr = minmax_abs_scalar.transform(x_tr)

        ## scale the test data based on training set parameters
        #
        x_test = minmax_abs_scalar.transform(x_test)

    ## end of if
    #


    ## perform XGBoost to do the feature importance analysis along with model training
    #
    ## Use the XGBoost classifier to understand the importance features
    ## This model is very stable and a perfect candidate for these types of
    ## unstructured datasets
    #

    xgboost = XGBClassifier(random_state=DEF_RAND_STATE)
    xgboost_grd_params = { 'booster' : [ 'gbtree', 'gblinear'] }

    ## gridsearch optimization
    #gd_sr_xgboost = GridSearchCV(estimator = xgboost, param_grid = xgboost_grd_params,
    #                            scoring = "accuracy", cv = StratifiedKFold(5))

    xgboost.fit(x_tr, y_tr)
    print(xgboost.feature_importances_)
    plt.bar(range(len(xgboost.feature_importances_)), xgboost.feature_importances_)

    ## save the feature importance analysis
    #
    xgboost_preeval_png = ofile.split(".txt")[0] + "_xgboost_tr" + ".png"
    plt.savefig(xgboost_preeval_png)
    xgboost_preeval_sorted_png = ofile.split(".txt")[0] + "_xgboost_tr_sorted" + ".png"
    plot_importance(xgboost)
    plt.savefig(xgboost_preeval_sorted_png)    

    ## decode and score the test set
    #
    xgboost_pred = xgboost.predict(x_test)
    xgboost_acc = accuracy_score(y_test, xgboost_pred)
    xgboost_rcl = recall_score(y_test, xgboost_pred)

    print ("XGBoost accuracy on the test set is: ", xgboost_acc)
    print ("XGBoost recall on the test set is: ", xgboost_rcl)

    return status

## end of method
#

# method: perform_rudimentary_analysis
#
# arguments: 
#  pd_dat_a: pandas data frame
#  target_events: list of events where analysis is to be performed
#  uniq_entries: dictionary containing list of uniq entries for columns
#  ofile_a: output file location
#
# return: 
#  None
#
## This method checks what fields/groups of sets had treatments. Patients taking treatments
## is assumed to have mental illness.
#
def perform_rudimentary_analysis(pd_dat_a, target_events, uniq_entries, ofile_a):

    ## label info as ground truth labels
    #
    treatment_info = pd_dat_a['treatment'].copy().to_numpy()

    ## loop through target events and assign labels
    #
    for ele in target_events:
        print ("processing ", ele)
        trgt_info = pd_dat_a[ele].copy().to_numpy()
        analysis_dict = separate_cols_for_labels(trgt_info, uniq_entries[ele], treatment_info)
        print(analysis_dict)

        ## separate Yes and No for and create two separate distributions
        #
        treatment_set, no_treatment_set = separate_classes(analysis_dict)
        save_plots(treatment_set, no_treatment_set, ele, ofile_a)
    ## end of for
    #
## end of method
#


# method: separate_classes
#
# arguments: 
#  analysis_dict: 2D dictionary with class separated entities
#
# return: 
#  treat_dct: subjects who receive treatment (label 0)
#  notreat_dct: subjects who receive no treatment (label 1)
#
## This method separate sets with treatments and no treatments into two!
#
def separate_classes(analysis_dict):

    ## initialize two sets for yes/no distributions
    #
    treat_dct = dict()
    notreat_dct = dict()

    ## iterate through analysis
    #
    for k,v in analysis_dict.items():
        treat_dct[k] = analysis_dict[k]['yes']
        notreat_dct[k] = analysis_dict[k]['no']

    ## end of for
    #

    ## return gracefully
    #
    return treat_dct, notreat_dct

## end of method
#


# method: save_plots
#
# arguments: 
#  treatment_set: list of label 0 members
#  no_treatment_set: list of label 1 members
#  fname_prefix: target name (i.e. somewhat, yes ,no, maybe)
#  ofile_a: output file for storing analysis
#
# return: 
#  None
#
## This method saves plots of the binary class based distribution
#
def save_plots(treatment_set, no_treatment_set, fname_prefix, ofile_a):

    plt.bar(treatment_set.keys(), treatment_set.values(), alpha = 0.5)
    plt.bar(no_treatment_set.keys(), no_treatment_set.values(), alpha = 0.5)
    plt.legend(['treatment', 'no_treament'])
    plt.title(fname_prefix)
    ## make sure the plot flag is active
    #
    if PLOT_F:
        plt.show()

    ## save in an output file
    #
    png_file = ofile_a.split(".txt")[0] + "_" + fname_prefix + ".png"
    print("saving file: ", png_file)
    plt.savefig(png_file)

## end of method
#

# method: separate_cols_for_labels
#
# arguments: 
#  col_info: a single column on pandas data frame 
#  uniq_entries: unique entries occuring on a single column (list)
#  lbl_info: column corresponding to the label
#
# return: 
#  odict: directory showing being treated or untreated entries
#
## This method tabulates what set of columns receives treatment. It's a Yes/No classification
#
def separate_cols_for_labels(col_info, uniq_entries, lbl_info):

    ## output directory
    #
    odict = dict()

    ## initialize the output directory
    #
    for ele in uniq_entries:
        odict[ele] = {'yes': 0, 'no': 0}
        
    ## calculate how many employees recieves treatment
    #
    for i in range(len(col_info)):

        if lbl_info[i] == 'yes':
            odict[col_info[i]]['yes'] += 1
        elif lbl_info[i] == 'no':
            odict[col_info[i]]['no'] += 1
        ## end of if
        #
    ## end of for
    #
    return odict
    
## end of method
#

# method: feature_importance_via_nn
#
# arguments: 
#  rand_seeds: list of random integers
#  x_tr_a: training data
#  y_tr_a: training labels
#  x_test_a: test data
#  y_test_a: test labels
#
# return: 
#  accs: overall accuracie per experiment
#  rcls: overall recall per experiment
#  imp_feat_acc: accuracies corresponding to feature dimensinons (list)
#  imp_feat_rcl: recalls corresponding to feature dimensinons (list)
#
## This method performance feature importance analysis by training NNs with different
## initialization and estimating it's dependences on specific feature dimensions.
## We zero padd specific feature dimensions to see how much reduction in performance
## it yields.
#
def feature_importance_via_nn(rand_seeds, x_tr_a, y_tr_a, x_test_a, y_test_a):

    ## initialize variables
    #
    accs = []
    rcls = []
    imp_feat_acc = []
    imp_feat_rcl = []    

    ## initialize with random seeds and train the model
    #
    for seed in rand_seeds:
        np.random.seed(seed)

        ## save performance in accuracy as well as recall
        ## accuracy alone would incorporate amount of error scores
        #
        feat_acc_analysis_dict = dict()
        feat_rcl_analysis_dict = dict()
        
        opt = SGD(lr=0.0009, decay=0, momentum=0.5, nesterov=True)
        mdl = Sequential([
            Dense(32, activation = 'relu'),
            Dense(32, activation = 'relu'),
            Dense(2, activation = 'softmax')])
        mdl.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        ## train on the train split
        #
        mdl.fit(x=x_tr_a, y=y_tr_a, batch_size = 32, epochs = 80, verbose = 0)
        print(mdl.summary())

        ## evaluate on the test set
        #
        mdl_score = mdl.predict(x_test_a).argmax(axis=-1)
        mdl_acc = accuracy_score(y_test_a.argmax(axis=-1), mdl_score)
        mdl_rcl = recall_score(y_test_a.argmax(axis=-1), mdl_score)
        print(mdl_acc)
        print(mdl_rcl)

        ## save the metric performance
        #
        accs.append(mdl_acc)
        rcls.append(mdl_rcl)

        ## perform zero padding on the feature dimensions and evaluate based on accuracy
        ## and recall
        #
        featdim_acc, featdim_rcl = zpad_test_feats_n_rescore(mdl, x_test_a, y_test_a)


        ## sort the values based on feature's importance
        ## low performance after zero padding means that feature had high contribution
        #
        sort_dict_based_on_values(featdim_acc, feat_acc_analysis_dict )
        sort_dict_based_on_values(featdim_rcl, feat_rcl_analysis_dict )

        ## save results 
        imp_feat_acc.append(feat_acc_analysis_dict.copy())
        imp_feat_rcl.append(feat_rcl_analysis_dict.copy())      
    ## end of for
    #


    ## return gracefully
    #
    return accs, rcls, imp_feat_acc, imp_feat_rcl

## end of method
#
        
# method: zpad_test_feats_n_rescore
#
# arguments: 
#  mdl_a: trained keras model
#  x_test_a: test features
#  y_test_a: test labels
#
# return: 
#  featdim_acc: accuracy corresponding to feature dimensions/idx
#  featdim_rcl: recall corresponding to feature dimensions/idx
#
## This method zero padds feature dimensions and rescores
#
def zpad_test_feats_n_rescore(mdl_a, x_test_a, y_test_a):    
    featdim_acc = dict()
    featdim_rcl = dict()    
    
    ## zero padd each column index and check model's performance on them
    #
    
    for i in range(len(x_test_a[0])):

        
        tmp_arr = x_test_a.copy()
        tmp_arr[:,i] = np.zeros((np.shape(tmp_arr)[0]), dtype=np.int32)

        mdl_score = mdl_a.predict(tmp_arr).argmax(axis=-1)
        mdl_acc = accuracy_score(y_test_a.argmax(axis=-1), mdl_score)

        ## round values to avoid floor noise
        #
        mdl_acc = round(mdl_acc, 3)
        mdl_rcl = recall_score(y_test_a.argmax(axis=-1), mdl_score)
        mdl_acc = round(mdl_rcl, 3)

        ## update performance values for each feature column
        featdim_acc[i] = mdl_acc
        featdim_rcl[i] = mdl_rcl

    ## end of for
    #

    ## return gracefully
    #
    return featdim_acc, featdim_rcl

## end of method
#

# method: sort_dict_based_on_values
#
# arguments: 
#  indict: dictionary with indexes as a key and performance as value
#  outdict: dictionary with indexes as a value and performance as a key
#
# return: 
#  None
#
## This method sorts dictionaries based on its values. Necessary to
## reorder the feature dimensions based on their importance
#
def sort_dict_based_on_values(indict, outdict):

    sorted_vals = sorted(list(indict.values()))

    key_list = list(indict.keys())
    val_list = list(indict.values())    

    ## loop through entries and flip the key,value pairs in a
    ## sorted mannaer
    #
    for i, (k, v) in enumerate(indict.items()):

        outdict[sorted_vals[i]] = val_list.index(sorted_vals[i])

    ## end of for
    #

    ## return gracefully
    #

## end of method
#


# method: summarize_top_n_feats
#
# arguments: 
#  featdim_list_a: list of dimension indices from the feature vector
#  params_a: names of columns related to feature indices
#  ofile_a: output file where results will be stored
#
# return: 
#  None
#
## This method writes the top N features in a file based on their
## contribution. Only Top 5 dimensions are loaded (for now...)
#
def summarize_top_n_feats(featdim_list_a, feat_names, ofile_a):

    ## transpose the list
    #
    np_arr = np.array(featdim_list_a)
    transposed_arr = np_arr.T
    transposed_featdim_list = transposed_arr.tolist()

    ## output file string prefix
    #
    out_str = ["feature dimensions which contribute the most to mental illness are: "]    

    ## loop through monte-carlo iterations
    #
    dim_in_order = []
    for ele in transposed_featdim_list:
        common_dims = Counter(ele)
        main_dim = common_dims.most_common(1)

        ## update the dimension of the main feature with the string for the output file
        #
        dim_in_order.append(main_dim)
        out_str.append(str(feat_names[main_dim[0][0]]) + "\t")

    ## end of for
    #
    print (out_str)
    write_ofile(ofile_a, out_str)
## end of method
#

# method: write_ofile
#
# arguments: 
#  fp_a: output file pointer
#  strlist: list of strings for the output file
#
# return: 
#  None
#
## This method logs the results: Feature which contributed the most
#
def write_ofile(fp_a, strlist):

    ## clean the file
    #
    fo = open(fp_a, "w").close()

    ## update entries line by line from the list of strings
    #
    with open(fp_a, "w") as fout:
        for line in strlist:
            fout.write(str(line))
            fout.write("\n")
    ## end of with
    #
## end of method
#

## end of file
#

    
    

