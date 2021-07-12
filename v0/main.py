#!/usr/bin/env python
#
# file: $(EXP)/python/src/main.py
#
# revision history:
#
# 20210525 (VS): initial version
#
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys
import argparse
import json

## import support modules
#
import analyze_dat as andat


#------------------------------------------------------------------------------
#
# global variables
#
#------------------------------------------------------------------------------

__FILE__ = os.path.basename(__file__)


#------------------------------------------------------------------------------
#
# the main program 
#
#------------------------------------------------------------------------------

# method: main
#
# arguments: stdin args
#
# return: none
#
# This function is the main program.
#
def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type = str)
    parser.add_argument('-o', '--odir', type = str)
    parser.add_argument('files', type = str, nargs = '*')
    args = parser.parse_args()

    ## collect arguments
    #
    if args.params is not None:
        pfile = args.params

    if args.odir is not None:
        odir = args.odir

    ## collect input (dat) file
    #
    dat_file = args.files[0]

    ## create output directory tree
    #
    create_dirtree(odir)
    
    ## load parameters
    #
    params = read_json(pfile)

    ## perform data analysis
    #
    if andat.run_analysis(dat_file, params, odir) == False:
        print("Error: (%s: %s): error processing data in method: (%s)" \
            %(__FILE__, __name__, "run_analysis"))
        exit(1)

    print("Analysis finished successfully...")

    exit(0)

## end of main
#



# method: create_dirtree
#
# arguments:
#  dirtree_a: heirarchy / tree structure of directory
#
# return:
#  None 
#
## create the directory tree 
#
def create_dirtree(dirtree_a):
    if not os.path.exists(dirtree_a):
        os.makedirs(dirtree_a)

## end of method
#


# method: read_json
#
# arguments:
#  f_a: input parameter file (JSON)
#
# return:
#  fcont: content of the file in dictionary
#
## reads the json parameter file
#
def read_json(f_a):

    with open(f_a, 'r') as fin:
        fcont = json.load(fin)


    ## return gracefully
    #
    return fcont

## end of method
#

## main program starts here
#
if __name__ == "__main__":
    main(sys.argv[:])

## end of file
#
