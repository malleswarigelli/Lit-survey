#Import numerical libraries
import sys
import numpy as np
import pandas as pd
import pickle

# Text preprocessing libraries
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

#Import resampling and modeling algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

#KFold CV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
warnings.filterwarnings('ignore')

# Access new data from Azure data store
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

print(pickle.format_version)

# Predict the class of ALL new/unseen Abstracts from Azure blob/container

# Access .json file from Azure container
keywords = sys.argv[1]

def access_json(keywords):
# function access .json file/Azure blob from Azure container using keywords
    import os
    from azure.storage.blob import BlobServiceClient
    from smart_open import open
    conn_str="DefaultEndpointsProtocol=https;AccountName=aimldatastore;AccountKey=hD7colWyUVASuZ2FOCZRudrPazQy7UOIEKhNkcsIu5q6cw9QKWDxMZczGkue8tCSXi0yAgxxbyXA+ASt9egFYw==;EndpointSuffix=core.windows.net"
    tparams = {
        'client': BlobServiceClient.from_connection_string(conn_str),
    }
    # file from Azure Blob Storage
    #print('python version:', sys.version)
    #print('pandas verion:', pd.__version__)

    # Try opening the file from azure. If that doesn't work, use the local json file
    try: # to open the file on azure
        f = open('azure://literaturemining/'+keywords+'.json', transport_params=tparams, encoding='utf-8')
        new_data = pd.read_json(f)
        f.close()
        print("We successfully opened the json file on azure")
    except: # open the local file
        print("We couldn't open the file on azure")
    #print(new_data.head())
    
    # processing the .json file
    # remove extra spaces, missing rows, empty strings in Abstract column and reset index and lemmatize words
    new_data['Abstract'] = new_data['Abstract'].str.strip()
    new_data= new_data[new_data['Abstract'].notna()]
    new_data= new_data[new_data['Abstract']!='']
    new_data= new_data.reset_index(drop=True)

    words = stopwords.words("english")
    from nltk.stem import WordNetLemmatizer
    lemmatizer=WordNetLemmatizer()

    new_data['lemma_text'] = new_data['Abstract'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(i) 
                            for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    data_set=list(new_data['lemma_text'])

     # Loading pickled tfidf vectorizer 
    vectorizer = pickle.load(open('vectorizer.pkl','rb'))
    test_input = vectorizer.transform(data_set)
    
   # loading picked model for predictions
    model = pickle.load(open('model.pkl','rb'))

    # predict class of new abstracts with DecisionTree classifier
    # create list of predicted labels
    DT_rank =[]
    for i in test_input:
        res=model.predict(i)
        #print(res)          
        if res=='high':
            DT_rank.append("high")
        elif res=='medium':
            DT_rank.append("medium")
        else:
            DT_rank.append("low")  
         

    # convert list of labels to dataframe
    DT_labels=pd.DataFrame({'DT_Labels': DT_rank})
    # concatenate dataframes
    Abstracts_class=pd.concat([new_data, DT_labels], axis=1)
    Abstracts_class['NLP_predicted_rank (1=high, 2=medium, 3=low)']= Abstracts_class['DT_Labels'].map({'high':1, 'medium':2, 'low':3})
    Abstracts_class.sort_values(by='NLP_predicted_rank (1=high, 2=medium, 3=low)', inplace=True)
    Abstracts_class = Abstracts_class.loc[:, ['Title','Abstract', 'OriginalURL','NLP_predicted_rank (1=high, 2=medium, 3=low)']]
    print(Abstracts_class.columns)
    # write resutls to 
    Abstracts_class.to_excel(''+keywords+'_output.xlsx', index=False)
    
    # Write NLP output to Azure blob storage container
    from azure.storage.blob import ContainerClient
    blob = BlobClient.from_connection_string(
        conn_str="DefaultEndpointsProtocol=https;AccountName=aimldatastore;AccountKey=hD7colWyUVASuZ2FOCZRudrPazQy7UOIEKhNkcsIu5q6cw9QKWDxMZczGkue8tCSXi0yAgxxbyXA+ASt9egFYw==;EndpointSuffix=core.windows.net",
        container_name="literaturemining",
        blob_name=''+keywords+'_output.xlsx')
    with open(''+keywords+'_output.xlsx', "rb") as data:
        blob.upload_blob(data, overwrite=True)

access_json(keywords)
    