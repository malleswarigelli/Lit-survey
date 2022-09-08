#Import numerical libraries
import sys
import numpy as np
import pandas as pd

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

# Researchers experiment data
df= pd.read_excel('Input1_.xlsx')
# rename colums and print head
df.rename(columns={'Title ': 'Title', 'rank of relevance (1=high, 2=medium, 3=low)':'Class'}, inplace=True)
#slice needed columns
df=df[['Title','Abstract','Class','Rank']]
# drop rows with no abstract, Class
df= df[df['Abstract'].notna()]
df=df[df['Class'].notna()]


# Text preprocessing: stemming Abstracts
stemmer1 = PorterStemmer()
words = stopwords.words("english")
df['stem_text'] = df['Abstract'].apply(
    lambda x: " ".join([stemmer1.stem(i) 
                        for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

# lemmatize abstracts
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
df['lemma_text'] = df['Abstract'].apply(
    lambda x: " ".join([lemmatizer.lemmatize(i) 
                        for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

## Slice feature and target variables, convert to arrays
data=df[['lemma_text','Rank']]
X= data['lemma_text'].values
y = data['Rank'].values

## Train test split data
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.33, random_state=42)

## Transform words into vectors with TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tf = vectorizer_tfidf.fit_transform(X_train.ravel())
X_train_tf = vectorizer_tfidf.transform(X_train.ravel())
    
#transforming test data into tf-idf matrix
X_test_tf = vectorizer_tfidf.transform(X_test.ravel())

# Supervised Machine Learning Classification models

# Fit naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
NB = MultinomialNB()
NB.fit(X_train_tf, y_train)    
#predicted train data
y_pred_train_NB = NB.predict(X_train_tf)
#print('Ranks of y_train\n:', y_train)
#print('-----------------------------------')
#print('NB classifier Training set predictions:\n',y_pred_train_NB)
#print('Training set accuracy:', accuracy_score(y_train, y_pred_train_NB))
y_pred_test_NB = NB.predict(X_test_tf)
#print('Ranks of y_test\n:', y_test)
#print('-----------------------------------')
#print('NB classifier Test set predictions:\n',y_pred_test_NB)
naive_score = accuracy_score(y_test, y_pred_test_NB)
#print('Test set accuracy:', np.round(naive_score,3))
#print(metrics.classification_report(y_test, y_pred_test_NB, target_names=['high', 'medium','low']))   
#print("Confusion matrix:")
#print(metrics.confusion_matrix(y_test, y_pred_test_NB))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train_tf, y_train)
#predicted train data
y_pred_train_LR = LR.predict(X_train_tf)
#print('Ranks of y_train\n:', y_train)
#print('-----------------------------------')
#print('LR classifier Training set predictions:\n',y_pred_train_LR)
#print('Training set accuracy:', accuracy_score(y_train, y_pred_train_LR))    
y_pred_test_LR = LR.predict(X_test_tf)
#print('Ranks of y_test\n:', y_test)
#print('-----------------------------------')
#print('LR classifier Test set predictions:\n',y_pred_test_LR)
LR_score = accuracy_score(y_test, y_pred_test_LR)
#print('Test set accuracy:', np.round(LR_score,3))

# Decision tree classifier
DT= DecisionTreeClassifier()
DT.fit(X_train_tf, y_train) 
    
#predicted train data
y_pred_train_DT = DT.predict(X_train_tf)
#print('Ranks of y_train\n:', y_train)
#print('-----------------------------------')
#print('DT classifier Training set predictions:\n',y_pred_train_DT)
#print('Training set accuracy:', accuracy_score(y_train, y_pred_train_DT))   

y_pred_test_DT = DT.predict(X_test_tf)
#print('Ranks of y_test\n:', y_test)
#print('-----------------------------------')
#print('DT classifier Test set predictions:\n',y_pred_test_DT)
DT_score = accuracy_score(y_test, y_pred_test_DT)
#print('Test set accuracy:', np.round(DT_score,3))

# SGDclassifier
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier(random_state=123)
SGD_clf_scores = cross_val_score(SGD, X_train_tf, y_train, cv=3)
#print('SGD cross val scores:', SGD_clf_scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (SGD_clf_scores.mean(), SGD_clf_scores.std() * 2))
SGD.fit(X_train_tf, y_train)
#predicted train data
y_pred_train_SGD = SGD.predict(X_train_tf)
#print('Ranks of y_train\n:', y_train)
#print('-----------------------------------')
#print('SGD classifier Training set predictions:\n', y_pred_train_SGD)
#print('SGD training set accuracy:', accuracy_score(y_train, y_pred_train_SGD))
y_pred_test_SGD = SGD.predict(X_test_tf)
#print('Ranks of y_test\n:', y_test)
#print('-----------------------------------')
#print('SGD classifier Test set predictions:\n',y_pred_test_SGD)
SGD_score = accuracy_score(y_test, y_pred_test_SGD)
#print('SGD Test set accuracy:', np.round(SGD_score,3))

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train_tf, y_train)
#predicted train data
y_pred_train_RF = RF.predict(X_train_tf)
#print('Ranks of y_train\n:', y_train)
#print('RF classifier Training set predictions:\n',y_pred_train_RF)
#print('RF training set accuracy:', accuracy_score(y_train, y_pred_train_RF))
#print('-----------------------------------')
y_pred_test_RF = RF.predict(X_test_tf)
#print('Ranks of y_test\n:', y_test)
#print('RF classifier Test set predictions:\n', y_pred_test_RF)
RF_score = accuracy_score(y_test, y_pred_test_RF)
#print('RF Test set accuracy:', np.round(RF_score,3))
#print('-----------------------------------')



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

    new_data['lemma_text'] = new_data['Abstract'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(i) 
                            for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    data_set=list(new_data['lemma_text'])
    # tranform text to vectorizer
    test_input = vectorizer_tfidf.transform(data_set)
    
    # predict class of new abstracts with DecisionTree classifier
    # create list of predicted labels
    DT_rank =[]
    for i in test_input:
        res=DT.predict(i)
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
    Abstracts_class.to_excel(''+keywords+'_output.xlsx')
    
    # Write NLP output to Azure blob storage container
    from azure.storage.blob import ContainerClient
    blob = BlobClient.from_connection_string(
        conn_str="DefaultEndpointsProtocol=https;AccountName=aimldatastore;AccountKey=hD7colWyUVASuZ2FOCZRudrPazQy7UOIEKhNkcsIu5q6cw9QKWDxMZczGkue8tCSXi0yAgxxbyXA+ASt9egFYw==;EndpointSuffix=core.windows.net",
        container_name="literaturemining",
        blob_name=''+keywords+'_output.xlsx')
    with open(''+keywords+'_output.xlsx', "rb") as data:
        blob.upload_blob(data, overwrite=True)

access_json(keywords)
    