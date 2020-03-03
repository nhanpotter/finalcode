import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
stop = stopwords.words('english')


def cleaning_dataset(input_file):
    df_train = pd.read_csv(input_file, encoding = 'unicode escape')
    
    #Pre-Processing...
    #convert all answers to string format...
    df_train['Ref Answer'] = df_train['Ref Answer'].astype(str)
    df_train['Answer'] = df_train['Answer'].astype(str)
    
    #convert all answers to lower case...
    df_train['Ref Answer'] = df_train['Ref Answer'].apply(lambda answer1: answer1.lower())
    df_train['Answer'] = df_train['Answer'].apply(lambda answer2: answer2.lower())
   
    #Remove of Stop Words from answers...
    df_train['Ref Answer'] = df_train['Ref Answer'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df_train['Answer'] = df_train['Answer'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
   
    
    return df_train


