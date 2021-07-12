Kaggle challenge (mental-health-in-tech-survey)

Directory Structure:

data/
Contains source csv file which will be used for training and testing

output/
Analysis and evaluation scores of the experiment are stored here.

params/
parameter files with parameters necessary for the data and models

v0 - v1/
Source scripts where the models are developed
v0/ contains a neural network based approach
v1/ contains XGBoost classifier. Other modules in both these directories are identical.


To run the experiment, goto v0 or v1 and run the run.sh script. i.e.
bash run.sh


Description:
We use treatment column as an indication that subject has a mental illness. This
quantity will be used as labels. Other columns are used as feature data.

The two approach used in this experiment focuses on one identical objective.
To find the underlying and most affecting features to the mental illness in the
workplace. This is a feature importance experiment. We analyze importance of the
features using all the entries/rows. But a small subset of data focusing on a
specific field/set can yield similar insight. For instance, we can only use
samples collected from the US and see what factors affect the domestic population's
mental health.

In v0, we develop a very small neural network MLP model. The goal isn't to
improve the performance of the classifier but to find important features which
correspond to (are correlated) with the mental illenss. We train and reinitialize
the NN model multiple times (similar to monte-carlo simulations). Each time the
model is trained, we use the same model and start zero-padding the feature
dimensions. Our model's performance takes a hit when a specific features are
unavailable. Model performing poorly when a specific feature is absent indicates
that the feature was important. These iterations are run N number of times e.g.
30-100. Feature indices with their scores are collected and sorted for each
iteration. At the end of the experiment, we select N most important features
observed from these collections. After running our model on the preprocessed
data, we find that NN model suggests that the following quantities carry the
most information:

self_employed	
seek_help	
gender	
anonymity	
no_employees	

In v1, we use a XGBoost classifier, which is robust hence ideal for unstructured
classification problems such as our problem. It uses ensemble method which is why it
will have more insight into balance between features. This types of model can
analyze the priority  of the feature instead of getting biased towards specific
dimensions/indices. We do feature analysis on the data using XGBoost model and 
find that following features carry the most information:

age
self_employed
no_employees
leave


Other simple analysis with bar plots can be found in the output directory.
These analysis files gives us insight into our data. For example, the data
for the subjects getting treatment and without treatment is balanced. There
are biases towards total employees based on gender or tech industry. Number
of employees has direct correlation with the mental illness.

