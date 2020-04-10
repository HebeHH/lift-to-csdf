#!/usr/bin/env python
# coding: utf-8

# In[392]:


import re
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import numpy as np


# # Success Rate

# In[543]:


labels = 'Could not generate CSDF',  'Success'
sizes = [2,27]
explode = 0,1


ff = {'family':'serif',
      'sans-serif':['Adobe Arabic'],
      'serif':['Times'],
      'size': 14
     }


plt.rc('font', **ff)

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',  startangle=90)
ax1.axis('equal') 

plt.title("CSDF Generation Success Rate", pad=20, 
          fontdict={'family':'sans-serif', 'size':18})
plt.savefig("Graphs/GenerationSuccessRatePie.png")


# In[642]:


with open("ufrecord.txt") as f:
    firstver = f.readlines()


# In[643]:


results = {"file" : [firstver[i] for i in range(0, len(firstver), 5)],
"thruput" : [float(re.findall(r'[0-9].+', firstver[i])[0]) for i in range(2, len(firstver), 5)],
"period" : [float(re.findall(r'[0-9].+', firstver[i])[0]) for i in range(3, len(firstver), 5)]}
results = pd.DataFrame(results)
results.thruput = pd.to_numeric(results.thruput)
results.period = pd.to_numeric(results.period)
results['name'] = [re.findall(r'/([a-zA-Z]+)', x)[0] for x in results.file]


# In[644]:


results = results[[True if '[' not  in x else False for x in results.file ]]

results['par'] = ['_par' in x for x in results.file]
results['rec'] = ['_rec' in x for x in results.file]
results['dz'] = ['_dz' in x for x in results.file]
results['cmap'] = ['_cmap_' in x for x in results.file]
results['svs'] = [int(re.findall(r'[0-9]+', x)[0]) for x in results.file]


# In[556]:


df = results[results.name == "mydot"]
df = df[~df.par]
ls = ['par', 'svs']
res = (df).merge(df, on = ls)
res['diffe'] = res.period_x/res.period_y
smsm = np.logical_and(res.dz_x == res.dz_y, res.cmap_x == res.cmap_y)
erk = np.logical_and(res.dz_x == False,  False == res.dz_y)
dz2 = np.logical_and(res.dz_x == True,  False == res.dz_y)
cmap2 = np.logical_and(res.cmap_x == False,  True == res.cmap_y)
the2s = np.logical_or(cmap2, dz2)
notrel = np.logical_or(smsm, erk)
notrel = np.logical_or(notrel, the2s)
ltd = ['period_x', "period_y", 'par', 'svs', 'dz_x', 'dz_y', 'cmap_x', 'cmap_y', 'diffe']

res[~notrel][ltd].head(40)


# If not composing maps, then should use arrayed zip (1-10% benefit, depending on size) because dearraying the zip will introduce some overhead with no chance of recovery.
# 
# If !cmap then !dz (bigger effect at smaller number) 
# If cmap then dz (bigger effect at bigger number)
# If dz then cmap
# If !dz then no difference

# In[657]:



results['ufs'] = [int(re.findall(r'[0-9]+', x)[1]) for x in results.file]
results.sort_values('svs')


# In[1]:


# results[~results.cmap]

# res = recres.merge(parres, left_on=['name'], right_on=['name'],
#           suffixes=('_recursive', '_parallel'))

def merge(var, df):
    ls = ['name', 'svs','ufs']
    res = (df[df[var]]).merge(df[~df[var]], on = ls,
          suffixes=('_'+var, '_not'+var))
    res['diffe'] = res['period_'+var] / res['period_not'+var]
    plt.hist(res.diffe, bins = 60)
    plt.plot()
    return res

k = merge('par', results)
# k[k.diffe == k.diffe.min()]
# [x for x in c if x != 1]
# k.groupby(['cmap']).diffe.min()
# k.sort_values(['svs'])


# In[658]:


colors =[ 'red', 'blue', 'green', 'indigo', 'yellow', 'orange', 'black',  'purple', 'violet', 'red', 'blue', 'green', 'indigo', 'yellow', 'orange', 'black',  'purple', 'violet','lime']
for i in k.ufs.unique():
    j=k[k.ufs==i].sort_values('diffe')
    plt.plot(j.diffe, j.svs, color=colors.pop())
plt.plot()


# In[648]:


k.groupby('ufs').diffe.max()-k.groupby('ufs').diffe.min()


# In[ ]:





# In[602]:


# results[~results.cmap]

# res = recres.merge(parres, left_on=['name'], right_on=['name'],
#           suffixes=('_recursive', '_parallel'))

def merge(var, df):
    ls = ['name', 'par', 'svs', 'cmap', 'dz']
    ls.remove(var)
    res = (df[df[var]]).merge(df[~df[var]], on = ls,
          suffixes=('_'+var, '_not'+var))
    res['diffe'] = res['period_'+var] / res['period_not'+var]
    plt.hist(res.diffe, bins = 60)
    plt.plot()
    return res

k = merge('par', results[results.name == 'mydotsmol'])
# k[k.diffe == k.diffe.min()]
# [x for x in c if x != 1]
# k.groupby(['cmap']).diffe.min()
# k.sort_values(['svs'])


# In[592]:



fig, ax = plt.subplots(figsize=(5,5))
# j = k[np.logical_and(k.dz == False, k.cmap == False)]
ax.scatter(j.svs, j.diffe)
# ax.legend()
# plt.plot()
plt.show()


# In[ ]:





# In[586]:


df = results[results.name == 'mydot']
df = df[df.rec]
ls = ['name', 'par', 'svs']
var = np.logical_and(df.dz == False, df.cmap == False)

k = (df[var]).merge(df[~var], on = ls,
      suffixes=('_ff', '_notff'))
k['diffe'] =  k['period_notff'] / k['period_ff'] 
# plt.hist(k.diffe, bins = 60)
# plt.plot()


# In[587]:



fig, ax = plt.subplots(figsize=(5,5))

# True, True
ax.scatter(k[np.logical_and(k.dz_notff == True, k.cmap_notff == True)].diffe, 
            k[np.logical_and(k.dz_notff == True, k.cmap_notff == True)].svs, color='red', label='both')
# False, True
ax.scatter(k[np.logical_and(k.dz_notff == False, k.cmap_notff == True)].diffe, 
            k[np.logical_and(k.dz_notff == False, k.cmap_notff == True)].svs, color='blue', label='cm')

# True, False
ax.scatter(k[np.logical_and(k.dz_notff == True, k.cmap_notff == False)].diffe, 
            k[np.logical_and(k.dz_notff == True, k.cmap_notff == False)].svs, color='green', label='dz')

ax.legend()
# plt.plot()
plt.show()


# In[573]:


j = results[results.name == 'addlisttoarray']
j = j[j.svs == 50]
j


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



plt.scatter(k[np.logical_and(k.dz == True, k.cmap == False)].diffe, 
            k[np.logical_and(k.dz == True, k.cmap == False)].svs, color='green')


# In[108]:


parres = results[['_par' in x for x in results.file]]
recres = results[['_notpar' in x for x in results.file]]


# In[109]:



parres


# In[111]:


res = recres.merge(parres, on=['name', 'svs'],
          suffixes=('_recursive', '_parallel'))
res


# In[13]:


res = recres.merge(parres, left_on='name', right_on='name',
          suffixes=('_recursive', '_parallel'))
def prettify_df(k):
    return k[['name'] + [c for c in k.columns if 'thruput' in c] + [c for c in k.columns if 'period' in c] ]

res = prettify_df(res)
res.

