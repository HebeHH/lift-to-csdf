#!/usr/bin/env python
# coding: utf-8

# In[28]:


from Generators import *

from os import listdir
from os.path import isfile, join
import sys
from tqdm import tqdm


# In[21]:


parreducemethods = [parallel_reduce, recursive_map, recursive_reduce]
parmethods = [parallel_reduce, parallel_map, recursive_reduce]
naivemethods = [recursive_map, recursive_reduce]
parmapmethods = [parallel_map, recursive_reduce]


# In[ ]:





# In[22]:


def generate_and_write(svs, fs):
    csdfs = []

    print("Generating")
    # make csdfs using parallel/not parallel map/reduce methods
    for sv in tqdm(svs):
        for fn in tqdm(fs):
            csdfs.append((fn, 'par'+str(sv), try_do("highLevel/"+fn, parmethods, [sv] * 10)))
            csdfs.append((fn, 'naive'+str(sv), try_do("highLevel/"+fn, naivemethods, [sv] * 10)))
            csdfs.append((fn, 'parreduce'+str(sv), try_do("highLevel/"+fn, parreducemethods, [sv] * 10)))
            csdfs.append((fn, 'parmap'+str(sv), try_do("highLevel/"+fn, parmapmethods, [sv] * 10)))   
    # delete ones where it didn't work
    csdfs = [x for x in csdfs if x[2] != False]         
    print('Generation Complete')

    # Transform by making zip 
    print("Discrete Zip")
    extras = []
    for x in tqdm(csdfs):
        if 'dz' not in x[1]:
            g = x[2]['graph']
            nodes = [getcn(n) for n in g['nodes']]
            if 'Zip' in nodes:
                csdf_copy = copy.deepcopy(g)
                discrete_zip(csdf_copy, None)
                extras.append((x[0],  x[1]+"_dz", csdf_copy))
    csdfs += extras
    csdfs = [(a[0], a[1], b) for a, b in dict([((x[0], x[1]), x[2]) for x in csdfs]).items()]
    print('Discrete Zip complete')


    # Write to file
    print("Writing to files")
    for csdf in tqdm(csdfs):
        g = csdf[2] if 'dz' in csdf[1] else csdf[2]['graph']
        nbase =  'curr/'+csdf[0]+"_"+csdf[1]
        n1 =nbase+'_.xml'
        n2 = nbase+'_cmap_.xml'
        write_csdf(g, n1)
        smushed = compose_maps(g)
        write_csdf(smushed, n2)

    print('Finished')


# In[23]:


mypath = 'highlevel/'
fs = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.endswith('json')]
fs = [x for x in fs if 'mm' not in x] + ['mmNT']


# In[30]:


if sys.argv[1] == 'all':
    generate_and_write([1,2,3,4,5,10,15,20,25,30,50,70,100], fs)
else:
    generate_and_write([3], ['mydotsmol'])


# In[ ]:




