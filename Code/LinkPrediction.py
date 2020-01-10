#!/usr/bin/env python
# coding: utf-8

# To run this notebook it is necessary to have installed the following packages:
# networkx and node2vec  

# In[1]:


#!pip install node2vec


# In[2]:


#from google.colab import drive
#drive.mount('/content/drive')


# # Imports & Initialization

# In[2]:


import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
import csv
import collections
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import pylab as pl
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


path = "./"
pathTrainingData = path + "training.txt"
pathTestData = path+"testing.txt"
#pathPages =path+'node_information/text/'
#pathPages =path+'data/'


# ## GRAPH CREATION

# In[3]:


G = nx.Graph() #undirected graph
GDi = nx.DiGraph() #directed graph
X=[]
Y=[]
X_Kaggle=[]

with open(pathTrainingData, "r") as f:
    for line in f:
        line = line.split()
        X.append([line[0], line[1]])
        Y.append(int(line[2]))
        if line[2]=='1':
            G.add_edge(line[0], line[1])
            GDi.add_edge(line[0], line[1])
        else:
            G.add_nodes_from([line[0],line[1]])
            GDi.add_nodes_from([line[0], line[1]])

with open(pathTestData, "r") as f:
    for line in f:
        line = line.split()
        X_Kaggle.append([line[0], line[1]])

for n in G.nodes:
    G.nodes[n]['community'] = 0

for n in GDi.nodes:
    GDi.nodes[n]['community'] = 0

X=np.array(X)
X_Kaggle=np.array(X_Kaggle)


# # Graph Information

# In[4]:


print(nx.info(GDi))
print('average clustering coefficient:  ' , nx.average_clustering(GDi))
#print('average shortest path length:  ' , nx.average_shortest_path_length(G))
degreesDi = [GDi.degree(n) for n in GDi.nodes()]
deg_hist=np.histogram(degreesDi, bins=[0,1, 2, 10, 20,30,40,50,60,70,80,90,100])
plt.hist(degreesDi, bins=[0,1, 2, 10, 20,30,40,50,60,70,80,90,100])
plt.title('Graph Degree')
plt.xlabel('Degree')
plt.ylabel('N° vertices')
plt.show()


# # Graph Feature Extraction

# In[7]:


# Features extraction methods

def jaccard(G,X):
    jaccardcoef=[]
    for i in range(X.shape[0]):
        try:
            coef = [[u, v, p]for u, v, p in nx.jaccard_coefficient(G, [(X[i][0], X[i][1])])][0]
            jaccardcoef.append(coef[2])
        except:
            jaccardcoef.append(0)
    return jaccardcoef

def adamic(G,X):
    adamicix=[]
    for i in range(X.shape[0]):
        try:
            coef = [[u, v, p]for u, v, p in nx.adamic_adar_index(G, [(X[i][0], X[i][1])])][0]
            adamicix.append(coef[2])
        except:
            adamicix.append(0)
    return adamicix

def preferentialAttachment(G,X):
    preferentialAtt=[]
    for i in range(X.shape[0]):
        try:
            coef = [[u, v, p]for u, v, p in nx.preferential_attachment(G, [(X[i][0], X[i][1])])][0]
            preferentialAtt.append(coef[2])
        except:
            preferentialAtt.append(0)
    return preferentialAtt

def resourceAllocation(G,X):
    resource_allocation=[]
    for i in range(X.shape[0]):
        try:
            coef = [[u, v, p]for u, v, p in nx.resource_allocation_index(G, [(X[i][0], X[i][1])])][0]
            resource_allocation.append(coef[2])
        except:
            resource_allocation.append(0)
    return resource_allocation

def soundarajan_hopcroft(G,X):
    soundarajan=[]
    for i in range(X.shape[0]):
        try:
            coef = [[u, v, p]for u, v, p in nx.cn_soundarajan_hopcroft(G, [(X[i][0], X[i][1])])][0]
            soundarajan.append(coef[2])
        except:
            soundarajan.append(0)
    return soundarajan

def commonNeighbors(G,X):
    commonN=[]
    for i in range(X.shape[0]):
        try:
            shortestArr =  nx.common_neighbors(G, X[i,0], X[i,1])
            commonN.append(len(sorted(shortestArr)))
        except:
            commonN.append(0)
    return commonN


# # Graph Embeddings

# In[8]:


def TrainGraphEmbeddings(GDi):
    #node2vec = Node2Vec(GDi, dimensions=16, walk_length=30, num_walks=200, workers=1)
    node2vec = Node2Vec(GDi, dimensions=4, walk_length=5, num_walks=5, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Save model for later use
    #model.save(path+"embeddingModeldim64Di.model")
    return model

def NodeSimilarity(NodePairs,model):
    node_sim=[]
    for nodepair in NodePairs:
        try:
            node_sim.append(model.wv.similarity(nodepair[0],nodepair[1]))
        except:
            node_sim.append(0.0)
    return node_sim


# # Text Feature Extraction
# Uncomment the codes of this section if the pathPages was specified and is correct

# In[9]:


#stop_words = open(path+'stopwords.txt','r').read().split(',') #Loading the prespecified stop words


# In[10]:


'''pages = []
for i in range(len(G.nodes)):
    f = open(pathPages+ str(i) + ".txt", encoding="utf8", errors='ignore')
    pages.append(re.sub('[0-9_]', '', f.read()))'''


# In[11]:


#vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop_words, min_df=0.00005) #TF-IDF model 


# In[12]:


#%time vectors_pages = vectorizer.fit_transform(pages) #Vectorizing pages


# In[13]:


#cos_sim_matrix = cosine_similarity(vectors_pages)#This operation may require a lot of RAM


# In[14]:


'''cos_sim = []
for e in X:
    idx1 = int(e[0])
    idx2 = int(e[1])
    cos_sim = cos_sim_matrix[idx1,idx2]
cos_sim = np.array(cos_sim)'''


# In[15]:


'''cos_sim_kaggle = []
for e in X_Kaggle:
    idx1 = int(e[0])
    idx2 = int(e[1])
    cos_sim_kaggle = cos_sim_matrix[idx1,idx2]
cos_sim_kaggle = np.array(cos_sim_kaggle)'''


# # Compute Features

# #### Training features

# In[16]:


cos_sim=[] #Loading of the precomputed cosine similarities of the training nodes pages(text)
with open(path+"cos_sim.txt", "r") as f:
    for line in f:
        line = line.split()
        cos_sim.append(float(line[0]))


# In[18]:


M=X.shape[0]
N= 8 # 7 extracted features
G_Features= np.zeros((M,N))

#model=TrainGraphEmbeddings(GDi)
model = Word2Vec.load(path+"embeddingModeldim32.model")
print('computing Jaccard...')
G_Features[:,0] = jaccard(G,X)
print('computing Adamic...')
G_Features[:,1] = adamic(G,X)
print('computing preferentialAtt...')
G_Features[:,2] = preferentialAttachment(G,X)
print('computing resource_allocation...')
G_Features[:,3] = resourceAllocation(G,X)
print('computing soundarajan...')
G_Features[:,4] = soundarajan_hopcroft(G,X)
print('computing commonNeighbors...')
G_Features[:,5] = commonNeighbors(G,X)
print('computing NodeSimilarity...')
G_Features[:,6] = NodeSimilarity(X, model)
print('computing cos_similarity_text...')
G_Features[:,7] = cos_sim
print('Finish')


# #### Kaggle Test features

# In[ ]:


cos_sim_kaggle=[] #Loading of the precomputed cosine similarities of the test nodes pages(text)
with open(path+"cos_sim_test.txt", "r") as f:
    for line in f:
        line = line.split()
        cos_sim_kaggle.append(float(line[0]))


# In[19]:


MKaggle=X_Kaggle.shape[0]
GKaggle_Features= np.zeros((MKaggle,N))
#model=TrainGraphEmbeddings(GDi)
model = Word2Vec.load(path+"embeddingModeldim32.model")
print('computing Jaccard...')
GKaggle_Features[:,0] = jaccard(G,X_Kaggle)
print('computing Adamic...')
GKaggle_Features[:,1] = adamic(G,X_Kaggle)
print('computing preferentialAtt...')
GKaggle_Features[:,2] = preferentialAttachment(G,X_Kaggle)
print('computing resource_allocation...')
GKaggle_Features[:,3] = resourceAllocation(G,X_Kaggle)
print('computing soundarajan...')
GKaggle_Features[:,4] = soundarajan_hopcroft(G,X_Kaggle)
print('computing commonNeighbors...')
GKaggle_Features[:,5] = commonNeighbors(G,X_Kaggle)
print('computing NodeSimilarity...')
GKaggle_Features[:,6] = NodeSimilarity(X_Kaggle, model)
print('computing cos_similarity...')
GKaggle_Features[:,7] = cos_sim_kaggle
print('Finish')


# #### Save Training Features

# In[20]:


pd.DataFrame(G_Features).to_csv(path+"XFeatures.csv", header=None, index=None)


# #### Save Kaggle Features

# In[21]:


pd.DataFrame(GKaggle_Features).to_csv(path+"XKaggleFeatures.csv", header=None, index=None)


# # Load saved features

# Load node2vec model

# In[22]:


model = Word2Vec.load(path+"embeddingModeldim64Di.model")


# Load Training Features

# In[23]:


G_Features=np.genfromtxt(path+"XFeatures.csv", delimiter=',')


# Load Kaggle Features

# In[24]:


GKaggle_Features=np.genfromtxt(path+"XKaggleFeatures.csv", delimiter=',')


# Split into Training and Testing

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(G_Features, Y, test_size=0.4, random_state=0)
print(X_train.shape)
print(X_test.shape)


# # Random Forest Optimizing hyperparameters

# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[27]:


n_estimators = [1, 2, 4, 8, 16, 32, 64]
train_results = []
test_results = []
for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# In[28]:


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# # Random Forest Model and Prediction

# In[29]:


clf = RandomForestClassifier(max_depth=9, n_estimators=15, random_state=0, )
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
FeatRandomAcc=(accuracy_score(y_test, y_pred))
FeatRandomF1=(f1_score(y_test, y_pred, pos_label=0))
print(FeatRandomAcc)
print(FeatRandomF1)


# In[ ]:





# # Feature Scaling

# In[30]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(G_Features)
Xs=scaler.transform(G_Features)
X_trains, X_tests, y_trains, y_tests = train_test_split(Xs, Y, test_size=0.4, random_state=0)


# In[31]:


scaler = MinMaxScaler()
scaler.fit(GKaggle_Features)
X_KaggleS=scaler.transform(GKaggle_Features)


# # Multilayer Perceptron

# getting correct parameters

# In[33]:


clfmlp = MLPClassifier(max_iter=150)
parameter_space = {
    'hidden_layer_sizes': [(7,7,5), (7,5)],
    'activation': ['tanh', 'relu'],
}
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(clfmlp, parameter_space, n_jobs=-1, scoring='f1' , cv=5, verbose=True)
clf.fit(X_trains, y_trains)


# In[34]:


# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# In[35]:


clfmlp = MLPClassifier(hidden_layer_sizes=(7,7,5), activation='tanh' ,max_iter=150 , random_state=3)
clfmlp.fit(X_trains,y_trains)
y_pred=clfmlp.predict(X_tests)
FeatMLPAcc=(accuracy_score(y_tests, y_pred))
FeatMLPF1=(f1_score(y_tests, y_pred, pos_label=0))
print(FeatMLPAcc)
print(FeatMLPF1)


# In[36]:


y_kaggle=clfmlp.predict(X_KaggleS)

y_kaggle = zip(range(len(y_kaggle)), y_kaggle)
# Write the output in the format required by Kaggle
with open(path+"predictionsnodesimMLPScaled150775.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in y_kaggle:
        csv_out.writerow(row) 


# # Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression 
Logclassifier = LogisticRegression(random_state = 0) 
Logclassifier.fit(X_trains, y_trains) 
y_predLog = Logclassifier.predict(X_tests) 
FeatLogAcc=(accuracy_score(y_tests, y_predLog))
FeatLogF1=(f1_score(y_tests, y_predLog, pos_label=0))
print(FeatLogAcc)
print(FeatLogF1)


# In[ ]:





# # HADAMARD

# In[38]:


model = Word2Vec.load(path+"embeddingModeldim64Di.model")


# In[39]:


Hadamard=[]
Hadamard_Kaggle=[]
empty=[]
for NodePair in X:
    try:
        Hadamard.append(model.wv.get_vector(NodePair[0])*model.wv.get_vector(NodePair[1]))
    except:
        empty.append[NodePair]

for NodePair in X_Kaggle:
    try:
        Hadamard_Kaggle.append(model.wv.get_vector(NodePair[0])*model.wv.get_vector(NodePair[1]))
    except:
        empty.append[NodePair]
            
Hadamard=np.array(Hadamard)
Hadamard_Kaggle=np.array(Hadamard_Kaggle)



# In[40]:


X_trainH, X_testH, y_trainH, y_testH = train_test_split(Hadamard, Y, test_size=0.4, random_state=0)


# ### Logistic HADAMARD
# 

# In[41]:


from sklearn.linear_model import LogisticRegression 
Logclassifier = LogisticRegression(random_state = 0) 
Logclassifier.fit(X_trainH, y_trainH) 
y_predLog = Logclassifier.predict(X_testH) 
HlogAcc=accuracy_score(y_testH, y_predLog)
HlogF1=f1_score(y_testH, y_predLog, pos_label=0)
print(HlogAcc)
print(HlogF1)


# ### MLP HADAMARD

# In[45]:


clfmlp = MLPClassifier(hidden_layer_sizes=(50,64), max_iter=100 , random_state=1)
clfmlp.fit(X_trainH,y_trainH)
y_pred=clfmlp.predict(X_testH)
HMLPAcc=accuracy_score(y_testH, y_pred)
HMLPF1=f1_score(y_testH, y_pred, pos_label=0)
print(HMLPAcc)
print(HMLPF1)


# ### Random Forest HADAMARD

# In[43]:


clfRandom = RandomForestClassifier()
parameter_space = {
    'max_depth': [5, 10, 20, 30],
    'n_estimators': [5, 10, 15],
}
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(clfRandom, parameter_space, n_jobs=-1, cv=3, verbose=True)
clf.fit(X_trainH, y_trainH)


# In[44]:


# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# In[46]:


clf = RandomForestClassifier(max_depth=20, n_estimators=15, random_state=0, )
clf.fit(X_trainH, y_trainH)

y_pred=clf.predict(X_testH)
HrandomAcc=accuracy_score(y_testH, y_pred)
HrandomF1=f1_score(y_testH, y_pred, pos_label=0)
print(HrandomF1)
print(HrandomAcc)


# # Comparaisons 

# In[49]:


# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [HlogAcc, HlogF1]
bars2 = [HMLPAcc, HMLPF1]
bars3 = [HrandomAcc, HrandomF1]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='LogisticR')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='MLP')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='RandomF')
 
# Add xticks on the middle of the group bars
plt.xlabel('Hadamard', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Acc', 'F1'])
plt.ylim(0, 1)
# Create legend & Show graphic
plt.legend()
plt.show()


# In[48]:


# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [FeatLogAcc, FeatLogF1]
bars2 = [FeatMLPAcc, FeatMLPF1]
bars3 = [FeatRandomAcc, FeatRandomF1]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='LogisticR')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='MLP')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='RandomF')
 
# Add xticks on the middle of the group bars
plt.xlabel('extracted features', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Acc', 'F1'])
 
plt.ylim(0, 1)
# Create legend & Show graphic
plt.legend()
plt.show()


# In[59]:


from sklearn.model_selection import validation_curve
train_scores, valid_scores = validation_curve(RandomForestClassifier(), G_Features, Y, param_name="max_depth", param_range=[4,6,8,10,15] ,scoring='f1', cv=5)


# In[60]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)
plt.plot([4,6,8,10,15], train_scores_mean, label="Training score",
             color="darkorange")
plt.plot([4,6,8,10,15], test_scores_mean, label="validation score",
             color="navy")
plt.legend()
plt.xlabel('Max depth')


# ### Validation MLP

# In[53]:


from sklearn.model_selection import validation_curve
train_scores, valid_scores = validation_curve(MLPClassifier(hidden_layer_sizes=(7,5)), Xs, Y, param_name="max_iter", param_range=[50,100,150,200] ,scoring='f1', cv=3)


# In[54]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)
plt.plot([50,100,150,200] , train_scores_mean, label="Training score",
             color="darkorange")
plt.plot([50,100,150,200] , test_scores_mean, label="validation score",
             color="navy")
plt.legend()
plt.xlabel('Max iterations')


# In[57]:


from sklearn.model_selection import validation_curve
train_scores, valid_scores = validation_curve(LogisticRegression(random_state = 0) , Xs, Y, param_name="C", param_range=[0.005,0.05,0.5,1, 1.5] ,scoring='f1', cv=3)


# In[58]:


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)
plt.plot([0.005,0.05,0.5,1, 1.5] , train_scores_mean, label="Training score",
             color="darkorange")
plt.plot([0.005,0.05,0.5,1, 1.5] , test_scores_mean, label="validation score",
             color="navy")
plt.legend()
plt.xlabel('Inverse of regularization')


# In[ ]:




