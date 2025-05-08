import numpy as np
import pandas as pd
import seaborn as sns
from loaddata import load_features
from sklearn.model_selection import train_test_split

ffilepath = input("Please enter the filepath to the female data: ")
mfilepath = input("Please enter the filepath to the male data: ")
fdata = load_features(ffilepath)
mdata = load_features(mfilepath)

#train-test split blabla



#Standard normalization blabla


#basic feature extraction blabla


#save features to csv file with y appended

