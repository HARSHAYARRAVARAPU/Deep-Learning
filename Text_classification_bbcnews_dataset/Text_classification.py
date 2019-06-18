
# coding: utf-8

# In[91]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd, xgboost, numpy as np, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


# In[92]:


df = pd.read_csv(r"D:\deep_learning\CNN\bbc-text.csv\bbc-text.csv")


# In[93]:


df.head()


# In[94]:


df['category_id'] = df['category'].factorize()[0]
df['category_id'].shape


# In[95]:


category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')


# In[96]:


category_id_df


# In[97]:


category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)


# In[98]:


df.groupby('category').category_id.count()


# In[99]:


tfidf = TfidfVectorizer(sublinear_tf =True, min_df =5, norm ='l2',encoding ='latin-1'
                       ,ngram_range =(1,2), stop_words ='english')


# In[100]:


features = tfidf.fit_transform(df.text).toarray()
labels = df.category_id


# In[101]:


print(features.shape)
print(labels.shape)


# In[102]:


sorted(category_to_id.items())


# In[109]:


#use chi square analysis to find the corelation between
#features (importance of words) and labels(categories)
from sklearn.feature_selection import chi2

N=3 #We are going to look for top 3 categories
#For each category, find words that are highly corelated to it
for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels==category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    
    print("'# '{}' :".format(category))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
    


# In[110]:


features_chi2


# In[113]:


from sklearn.manifold import TSNE
SAMPLE_SIZE = int(len(features)*0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size =SAMPLE_SIZE, replace =False)


# In[115]:


projected_features = TSNE(n_components =2, random_state =0).fit_transform(features[indices])


# In[116]:


type(projected_features)


# In[117]:


my_id =0
projected_features[(labels[indices] == my_id).values]


# In[118]:


colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']

# Find points belonging to each category and plot them
for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
          fontdict=dict(fontsize=15))
plt.legend()


# In[121]:


from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

models = [RandomForestClassifier(n_estimators =200, max_depth =3, 
                                random_state= 42),
         MultinomialNB(),
         LogisticRegression(random_state=42),]


# In[125]:


CV =5

cv_df = pd.DataFrame(index =range(cv*len(models)))
entries =[]


# In[129]:


for model in models:
    model_name = model.__class__.__name__
    
    accuracies = cross_val_score(model, features, labels, 
                                 scoring ='accuracy',cv=CV)
    
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))


# In[134]:


cv_df =pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[135]:


cv_df


# In[139]:


import seaborn as sns

#sns.boxplot(x='model_name', y ='accuracy', data ='cv_df')
sns.stripplot(x='model_name', y = 'accuracy', data ='cv_df',
            size =8, jitter=True, edgecolor ="blue", linewidth =2)


# In[142]:


cv_df.groupby('model_name').accuracy.mean()


# In[144]:


from sklearn.model_selection import train_test_split
model = LogisticRegression()
X_train, X_test, y_train,y_test, indices_train, indices_test = train_test_split(features,
                                                   labels, df.index, test_size =0.33, random_state =42)


# In[151]:


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)


# In[152]:


y_pred_proba


# In[154]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
#sns.heatmap(conf_mat, annot=True, fmt='d',
#            xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')

