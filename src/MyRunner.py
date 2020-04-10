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


# In[22]:


def generate_and_write_depreciated(svs, fs):
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


# In[33]:


def write_all(fn, method_name, method_list, sv, path = "highLevel/"):
    name = "curr/" + fn + str(sv) + "_" + method_name
    csdf = try_do(path+fn, method_list, [sv] * 10)

    if csdf != False:
        g = csdf['graph']
        write_csdf(g, name + "_.xml")

        g_copy = copy.deepcopy(g)
        g_copy = discrete_zip(g_copy, None)
        write_csdf(g_copy, name + "_dz_.xml")

        g = compose_maps(g)
        write_csdf(g, name + "_cmap_.xml")

        g = discrete_zip(g, None)
        write_csdf(g, name + "_dz_cmap_.xml")

def generate_and_write(svs, fs):
    for filename in fs:
        print("Writing", filename)
        for sv in tqdm(svs):
            write_all(filename, "par", parmethods, sv)
            write_all(filename, "naive", naivemethods, sv)
            write_all(filename, "parreduce", parreducemethods, sv)
            write_all(filename, "parmap", parmapmethods, sv)


# In[23]:


mypath = 'highLevel/'
fs = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.endswith('json')]
fs = [x for x in fs if 'mm' not in x] + ['mmNT']


# In[34]:


if sys.argv[1] == 'all':
    generate_and_write([1,2,3,4,5,10,15,20,25,30,50,70,100], fs)
elif sys.argv[1] == "test":
    generate_and_write([3], ['mydotsmol'])
else: 
    ls = [int(x) for x in sys.argv[1].split(',')]
    if len(sys.argv) > 2:
        fs = [sys.argv[2]]
    generate_and_write(ls, fs)


# In[ ]:




