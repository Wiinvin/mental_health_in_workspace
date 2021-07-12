#!/usr/bin/env python
#
# file: $(EXP)/python/src/preprocess.py
#
# revision history:
#
# 20210525 (VS): initial version
#
#------------------------------------------------------------------------------

# import system modules
#
import pandas as pd
import numpy as np


## global variables
#
LOWER_AGE_LIMIT = 1
UPPER_AGE_LIMIT = 120



## This class preprocesses the data collected from the Pandas dataframe
## The class offers generic/modular methods to modify data. Data speciic
## modification should be done in the source script
#
class Preprocess():

    # method: __init__ 
    #
    # arguments:
    #  pd_dat_a: pandas data frame
    #  params_a: parameters as a dictionary
    #  ignore_labels_a: list of string/labels/headers to be ignored
    #
    # return:
    #  None
    #
    ## This method is a constructor. It initializes the variables for the
    ## class and cleans up some data.
    #
    def __init__(self, pd_dat_a, params_a, ignore_labels_a):

        self.headers_d = pd_dat_a.columns
        self.n_entries_d, self.n_columns_d = pd_dat_a.shape
        self.ignore_labels_d = ignore_labels_a
        
        ## clean data by converting all the letters/characters to small letters
        #
        self.clean_dat(pd_dat_a)


    # method: collect_unique_values
    #
    # arguments:
    #  pd_dat_a: pandas data frame
    #  uniq_entries_a: dictionary with list of uniq values for each column of pdframe
    #
    # return:
    #  None
    #
    ## this method generates list of unique values seen for each column
    #
    def collect_unique_values(self, pd_dat_a, uniq_entries_a):

        for col_lbl in pd_dat_a.columns:

            if col_lbl not in self.ignore_labels_d:

                uniq_entries_a[col_lbl] = pd_dat_a[col_lbl].unique()

            ## end of if
            #
        ## end of for
        #
    ## end of method
    #

    
    # method: clean_dat
    #
    # arguments:
    #  pd_dat_a: pandas data frame
    #
    # return:
    #  None
    #
    ## This method converts all the ASCII strings to lower letter
    #
    def clean_dat(self, pd_dat_a):

        uniq_entries = dict() ## temporary. just for analysis purposes
        
        ## iterate through all the columns
        #
        for col_lbl in pd_dat_a.columns:

            ## convert string to lowercase
            #
            pd_dat_a.rename({col_lbl : col_lbl.lower()}, inplace = True, axis = 1)

        ## end of for
        #

        ## make all the entries lowercase first
        #
        self.lowercase_entries(pd_dat_a)

        self.collect_unique_values(pd_dat_a, uniq_entries) ## temporary. just for analysis purposes
        #print(uniq_entries) ## temporary. just for analysis purposes

    ## end of method
    #


    # method: assign_ids2strs
    #
    # arguments:
    #  uniq_entries_a: dictionary with list of uniq values for each column of pdframe
    #  str_ids_a: ids associated with uniq entries of columns
    #  ignore_col: do not process columns listed in this argument
    #
    # return:
    #  None
    #
    ## assign integer ids to all the uniq values observed on each column
    #
    def assign_ids2strs(self, uniq_entries_a, str_ids_a, ignore_col = []):

        for k in uniq_entries_a.keys():

            if k in ignore_col:
                continue
        
            str_ids_a[k] = dict()

            for i,v in enumerate(uniq_entries_a[k]):

                str_ids_a[k][v] = i

            ## end of for 2
            #
        ## end of for 1
        #
    ## end of method
    #

    
    # method: flag_nans
    #
    # arguments:
    #  pd_dat_a: pandas dataframe
    #  idx_list: list of ids corresponding to row for Nan values
    #
    # return:
    #  None
    #
    ## flag rows with missing values and return the indices
    #
    def flag_nans(self, pd_dat_a, idx_list):

        ## find the null values
        #
        nan_flag_pd = pd_dat_a.isnull()

        ## iterating over rows is very inefficient in pandas
        ## instead convert it to boolean numpy array and make iterations
        ## faster
        #
        nan_flag_np = nan_flag_pd.to_numpy()
        
        for i in range(len(nan_flag_np)):
            for j in range(len(nan_flag_np[i])):
                if nan_flag_np[i][j] == True:
                    idx_list.append(i)
                    break
        #print(idx_list)

        ## return gracefully
        #
    ## end of method
    #

    # method: rm_entries
    #
    # arguments:
    #  pd_dat_a: pandas dataframe
    #  idx: id corresponding to row for Nan values
    #
    # return:
    #  None
    #
    ## This method removes the indices passed from a list
    #
    def rm_entries(self, pd_dat_a, idx):

        rows = pd_dat_a.index[idx]
        pd_dat_a.drop(rows, inplace = True)

    ## end of method
    #
    

    
    # method: lowercase_entries
    #
    # arguments:
    #  pd_dat_a: pandas dataframe
    #
    # return:
    #  None
    #
    ## This method converts all the str/characters to lowercase for consistancy
    #
    def lowercase_entries(self, pd_dat_a):

        ## iterate through each column
        #
        for col_lbl in pd_dat_a.columns:


            ## check the first entry to make sure its not a numeric
            #
            if not isinstance(pd_dat_a[col_lbl][0], (int, float, np.int64)):  ## generalize this later

                pd_dat_a[col_lbl] = pd_dat_a[col_lbl].str.lower()

            ## end of if
            #

        ## end of for
        #
    ## end of method
    #

        
    ## TODO: if interested in time-series analysis
    #
    def parse_timestamps(self):
        pass


## end of class
#



# method: update_ages
#
# arguments:
#  pd_dat_a: pandas dataframe
#  use_mean: inputation with mean values
#
# return:
#  None
#
## This method converts all the str/characters to lowercase for consistancy
#
def update_ages(pd_dat_a, use_mean = False):

    ## collect the target frame
    #
    age_col_a = pd_dat_a['age']

    if use_mean:
        mean_age = age_col_a.mean()


    for i in range(len(age_col_a)):

        if age_col_a[i] < LOWER_AGE_LIMIT or age_col_a[i] > UPPER_AGE_LIMIT:

            if use_mean:
                pd_dat_a.loc[i, "age"] = mean_age
            else:
                pd_dat_a.loc[i, "age"] = np.nan

        ## end of if
        #
    ## end of for
    #
## end of method
#

# method: update_genders
#
# arguments:
#  pd_dat_a: pandas dataframe
#
# return:
#  None
#
## This method makes limited number of entries in genders and filters
## text entries
#
def update_genders(pd_dat_a):

    gender_col_a = pd_dat_a['gender']

    for i in range(len(gender_col_a)):

        if gender_col_a[i] == 'm' or gender_col_a[i].startswith("ma"):
            pd_dat_a.loc[i, "gender"] = "male"

        elif gender_col_a[i] == 'f' or gender_col_a[i].startswith("fe"):
            pd_dat_a.loc[i, "gender"] = "female"
            
        else:
            pd_dat_a.loc[i, "gender"] = "lgbtq"

        ## end of if/else
        #
    ## end of for
    #
## end of method
#


# method: gen_ordinal_feats_labs
#
# arguments:
#  pd_dat_a: pandas dataframe
#  str_ida_a: ids related to string of the column entries
#  lbl_list:
#  numeric_cols: index at which numeric columns are found
#
# return:
#  dat_arr: data array numpy
#  lbl_arr: label array numpy
#  feat_names: corresponding names of the feature dimensions
#
#
## TODO: Apply better logic to combine 'treatment' and 'work_interfere' values
#
def gen_ordinal_feats_labs(pd_dat_a, str_ids_a, lbl_list, numeric_cols):


    lbl_arr = pd_dat_a[lbl_list[0]].copy()  ## just one entry for now...
    pd_dat_a.drop(columns = lbl_list, inplace=True)
    
    dat_arr = np.zeros(pd_dat_a.shape, dtype=np.int32)
    feat_names = pd_dat_a.columns
    
    for i, col_lbl in enumerate(pd_dat_a.columns):

        if col_lbl in numeric_cols:
            dat_arr[:,i] = pd_dat_a[col_lbl].copy().to_numpy()
            continue
        
        tmp = pd_dat_a[col_lbl].copy()
        id_arr = replace_str_w_ids(tmp, str_ids_a)

        dat_arr[:,i] = id_arr
    
    #np_dat_a = pd_dat_a.to_numpy()
    lbl_arr = replace_str_w_ids(lbl_arr, str_ids_a)

    return dat_arr, lbl_arr, feat_names


# method: replace_str_w_ids
#
# arguments:
#  col_info_a: column info of a pandas frame
#  str_ids_a: ids related to string of the column entries
#
# return:
#  col_np_arr: column info as a numpy array
#
#
## This is also known as ordinal Encoding of data
#
def replace_str_w_ids(col_info_a, str_ids_a):


    col_np_arr = np.zeros((len(col_info_a)), dtype=np.int32)
    
    target_key = col_info_a.name
    target_ids = str_ids_a[target_key]

    tmp_arr = col_info_a.to_numpy()

    for i in range(len(tmp_arr)):
        col_np_arr[i] = target_ids[tmp_arr[i]]

    return col_np_arr


# method: gen_onehot_feats_labs
#
# arguments:
#  pd_dat_a: pandas dataframe
#  str_ida_a: ids related to string of the column entries
#  lbl_list:
#  numeric_cols: index at which numeric columns are found
#
# return:
#  dat_arr: data array numpy
#  lbl_arr: label array numpy
#  feat_names: corresponding names of the feature dimensions
#
#
## This method creates features (elements collected from unstrucutred data)
## into one hot encoded vectors. Onehot feature should be used carefully such
## that resultant feature vector isn't too wide such that ML models can't
## process it.
#
def gen_onehot_feats_labs(pd_dat_a, str_ids_a, lbl_list, numeric_cols):

    
    lbl_arr = pd_dat_a[lbl_list[0]].copy()  ## just one entry for now...
    pd_dat_a.drop(columns = lbl_list, inplace=True)
    
    feat_names = pd_dat_a.columns

    dummy_dat = pd_dat_a.copy()

    for i, col_lbl in enumerate(dummy_dat.columns):
        
        if col_lbl in numeric_cols:
            continue


        dummies = pd.get_dummies(dummy_dat[col_lbl])

        dummy_dat = pd.concat([dummy_dat, dummies], axis = 'columns')

        dummy_dat.drop(col_lbl, axis = 'columns', inplace = True)

    ## end of for
    #
    
    dat_arr = dummy_dat.copy().to_numpy()
    lbl_arr = replace_str_w_ids(lbl_arr, str_ids_a)

    ## return gracefully
    #
    return dat_arr, lbl_arr, feat_names

## end of method
#


## end of file
#






    
