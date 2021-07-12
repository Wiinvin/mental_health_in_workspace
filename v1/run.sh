#!/bin/bash

# define global locations:
#  this script sets up some general environment variables corresponding
#  to various locations the in the experiment directory
#
PARAMS="../params/v1_params.json"
OUTPUT="../output/v1_out_xgb"
FILES="../data/survey.csv"

## run the program
#
python main.py -p $PARAMS -o $OUTPUT $FILES

## end of file
#
