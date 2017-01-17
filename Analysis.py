
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np


# The original downloaded dataset has 6,647,235 rows (calculated with wc -l). I will thus begin exploring a smaller dataset.

# In[3]:

df = pd.read_table('sgadata_costanzo2009_rawdata_101120.txt', header = None)


# In[4]:

df.head()


# In[5]:

df.tail()


# What is this dataset? There are 12 columns in the dataset. 
# 
# #### Description of Genes
# 
# Query ORF 
# 
# Query gene name
# 
# Array ORF
# 
# Array gene name
# 
# 
# #### Quantizing the Interaction
# 
# Genetic interaction score (ε)
# 
# Standard deviation
# 
# p-value
# 
# Query single mutant fitness (SMF)
# 
# Query SMF standard deviation
# 
# Array SMF
# 
# Array SMF standard deviation
# 
# Double mutant fitness
# 
# Double mutant fitness standard deviation

# ### Preprocessing

# In[6]:

len(df)


# In[7]:

1712 * 3885 


# In[8]:

df.head()


# In[9]:

df2 = df.ix[:, [0,2,4,5,6]]


# In[10]:

df2.columns = ["QueryORF", "ArrayORF", "Score", "Std-Dev", "P-Val"]
df2.head()


# In[11]:

df3 = df2.dropna(subset = ["Score"])


# In[12]:

df3.head()


# In[13]:

df3.info()


# In[14]:

len(df3["QueryORF"].unique())


# In[92]:

list(df3["QueryORF"].unique())


# In[93]:

list(df3["QueryORF"].unique()).index('YAL025C_damp')


# In[15]:

len(df3["ArrayORF"].unique())


# ## Task 1

# Implementation Steps:
# 
# (1) Create Gene Interaction Matrix
# 
# (2) Compute PCC between each pair of genes (worry about averaging later)
# 
# (3) Screen those that have PCC > .2
# 
# (4) Graph interactions potentially with Cytoscape

# To generate the network shown in Fig. 1, genetic interaction profile similarities were measured for all query and array gene pairs by computing Pearson correlation coefficients (PCC) from the complete genetic interaction matrix. Correlation coefficients of gene pairs screened both as queries and as arrays were averaged. Gene pairs whose profile similarity exceeded a PCC > 0.2 threshold were connected in the network, and an edge-weighted spring-embedded layout, implemented in Cytoscape (S4), was applied to determine node position. Genes sharing similar patterns of genetic interactions located proximal to each other in two-dimensional space, while less-similar genes were positioned further apart.

# In[73]:

df3.head()


# In[83]:

df5 = df3.ix[:, 0:3]
df5.head()


# We generate a mapping now from numbers to ORFs, so we can generate the adjacency matrix and later refer back to the genes represented.

# In[84]:

df5.tail()


# In[94]:

queryLabels = list(df5["QueryORF"].unique())
arrayLabels = list(df5["ArrayORF"].unique())


# In[103]:

tmp = arrayLabels + queryLabels


# In[104]:

len(tmp)


# In[106]:

len(set(tmp))


# In[96]:

def queryID(x):
    return queryLabels.index(x)

def arrayID(x):
    return arrayLabels.index(x)
    

df5["QueryID"] = df5["QueryORF"].map(queryID)
df5["ArrayID"] = df5["ArrayORF"].map(arrayID)


# In[98]:

df5.head(50)


# In[142]:

len(df5["QueryID"].unique())


# In[126]:

edge_list = df5[["QueryID", "ArrayID", "Score"]]


# In[132]:

edge_list.head()


# ### Code for the generation of an adjacency matrix

# In[151]:

# import numpy as np
def edgesToAdjMatrix(edge_list):
    import scipy.sparse as sps
    A = np.array(edge_list.values.tolist())
    i, j, weight = A[:,0], A[:,1], A[:,2]
    # find the dimension of the weight matrix
    dimI =  len(set(i))
    dimJ =  len(set(j))

    B = sps.lil_matrix((dimI, dimJ))
    for i,j,w in zip(i,j,weight):
        B[i,j] = w
    return B


# In[148]:

adj = edgesToAdjMatrix(edge_list)


# In[176]:

adj2.get_shape()[0]


# In[152]:

adj2 = edgesToAdjMatrix(edge_list)


# In[156]:

import pickle
with open('qcb.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([df5, edge_list, adj, adj2], f)


# In[180]:

#Calculate pearson similarity matrix (sparse) from the adjacency sparse matrix
# from scipy.stats.stats import pearsonr
# import scipy.sparse as sps
def adjToPearsonRows(adj):
    nrows = adj.get_shape()[0]
    C = sps.lil_matrix((nrows,nrows))
    for i in range(0,nrows):
        for j in range(i+1,nrows):
            row1 = adj.getrow(i).toarray()[0]
            row2 = adj.getrow(j).toarray()[0]
            sim_score = pearsonr(row1,row2)[0]
            if (sim_score > .2):
                C[i,j] = sim_score
                C[j,i] = sim_score
            
    return C


# In[181]:

simQ = adjToPearsonRows(adj2) #pearson similarity between all query mutants
simA = adjToPearsonRows(adj2.transpose()) #pearson similarity between all array mutants


# ## Code to get the degrees across all genes in the network

# In[215]:

ct = [0]*1711
for i in range(1711):
    ct[i] = np.count_nonzero(simQ.getrow(i).toarray()[0])
    print(ct[i])

ct1 = [0]*3885
for i in range(3885):
    ct1[i] = np.count_nonzero(simA.getrow(i).toarray()[0])

ctFreq = ct + ct1


# In[243]:

n, bins, patches = plt.hist(ctFreq, normed=True, bins=30)
plt.style.use('ggplot')
plt.xlabel('Degree Distribution')
plt.ylabel('Fraction of Genes');


# ### Code to grab all of the similarity scores

# In[218]:

ct2 = []
for i in range(1711):
    for j in range(i+1, 1711):
        if simQ[i,j] > .2:
            ct2.append(simQ[i,j])

ct3 = []
for i in range(3885):
    for j in range(i+1, 3885):
        if simQ[i,j] > .2:
            ct3.append(simA[i,j])

ctSim = ct2 + ct3


# In[274]:

plt.hist(ctSim, normed = True, bins=30)
plt.xlabel('Similarity Score Distribution')
plt.ylabel('Percent of Genes')


# # Task 2

# We measured the number of positive and negative interactions for all 3885 non-essential array deletion mutants at the intermediate cutoff (|"| > 0.08, p < 0.05). Genetic interaction hubs were selected as the top 10% highest connected genes. Genes with a bias towards positive interactions were selected by finding genes with at least 30 total interactions and positive to negative ratio greater than 1, which is twice the background ratio. Genes with a bias towards negative interactions were selected by finding genes with at least 30 total interactions and positive to negative ratio lower than 0.25, which is one-half of the background ratio. Both of these sets consisted of approximately 130 genes.
# 

# Implementation Steps:
# 
# (1) For each of the 3885 "non-essential array deletion mutants at the intermediate cutoff", get the number of positive and negative interactions. 
# 
# (2) Mark interaction hubs as the ones that are in the 10th percentile in terms of these interactions?
# 
# (3) Generate positive bias genes according to afformentioend criteria
# 
# (4) Generate negative bias genes according to aforementioned criteria. 

# In[16]:

df4 = df3[(df3["P-Val"] < .05) & (abs(df3["Score"]) > .08) ]


# In[17]:

len(df4)


# In[18]:

def pos(x):
    return int(x > 0)

def neg(x):
    return int(x < 0)
    

df4["Pos"] = df4["Score"].map(pos)
df4["Neg"] = df4["Score"].map(neg)


# In[19]:

df4.head()


# In[20]:

df4[["Pos", "Neg"]].sum()


# In[37]:

pos = df4.groupby("ArrayORF")["Pos"].sum()
pos


# In[38]:

neg = df4.groupby("ArrayORF")["Neg"].sum()
neg


# In[39]:

tot = df4.groupby("ArrayORF").size()
tot


# In[40]:

len(tot)


# In[41]:

len(pos)


# In[42]:

len(neg)


# In[43]:

type(pos)


# In[45]:

ratDf = pd.DataFrame({"Pos": pos, "Neg": neg, "Tot": tot})


# In[293]:

ratDf["ratio"] = ratDf["Pos"] / (ratDf["Tot"] + 1)


# In[298]:

ratDf["ratio"].plot(kind = 'hist', x = "Ratio", bins = 15, xlim = (0,1 ))


# In[282]:

list(ratDf["ratio"])


# In[72]:

ratDf["Tot"].plot(kind = 'hist')


# In[81]:

ratDf.head()


# In[109]:

ratDf["Neg"].plot(kind = 'hist', xlim = (0, 250))


# In[108]:

ratDf["Pos"].plot(kind = 'hist')


# In[110]:

totalInteractions = ratDf["Tot"].sum()


# In[113]:

ratDf["Neg%"] = ratDf["Neg"]/totalInteractions


# In[116]:

ratDf["Pos%"] = ratDf["Pos"]/totalInteractions


# In[118]:

ratDf["Pos%"].plot(kind = 'hist')


# In[312]:

dat, the_bins, patches = plt.hist(list(ratDf["Pos"]), normed = False, bins=10)


# In[314]:

dat2, the_bins2, patches2 = plt.hist(list(ratDf["Neg"]), normed = False, bins = the_bins)


# In[329]:

with open('bar.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([dat, dat2, the_bins], f)

