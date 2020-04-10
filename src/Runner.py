#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Generators import *

from os import listdir
from os.path import isfile, join
from tqdm import tqdm


# In[ ]:


fs = """asum                           gemvN                                   mmTT                         
 blackScholes                   gemvT                          mandelbrot                     molecularDynamics              mv                             scal
  gesummvNN                             mvAsMM   addlisttoarray
 dot                            gesummvTT                      mmNN              mappy             mriqComputeQ                   nbody
               mmNT              gesummvttfull        gesummvnnfull
                 kmeans                         mmTN      mydotsmol  mydot                     mriqPhiMag                     nearestNeighbour"""
fs = [x.strip() for x in fs.split(' ') if len(x) > 1]
fs


# In[ ]:



mypath = 'highlevel/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.endswith('json')]


# In[ ]:


len(onlyfiles)


# In[ ]:


recmethods = [discrete_zip, parallel_map, recursive_reduce]
printgraph_simple(compose_maps( try_do("highLevel/mydot", recmethods, [3] * 10)['graph']))


# In[ ]:


k = compose_maps( try_do("highLevel/addlisttoarray", recmethods, [3] * 10)['graph'])
# write_csdf(k, 'csdf_xmls/mdtpar.xml')


# In[ ]:


len(fs)


# In[3]:


csdfs = []
names = []


# In[6]:


# try_do("highLevel/mydot", parmethods, [sv] * 10)
mypath = 'highlevel/'
fs = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.endswith('json')]
fs = [x for x in fs if 'mm' not in x] + ['mmNT']
fs = ['mmNT']
# csdfs = []
# done: [3, 5, 7, 10, 15, 50]
parreducemethods = [parallel_reduce, recursive_map, recursive_reduce]
parmethods = [parallel_reduce, parallel_map, recursive_reduce]
naivemethods = [recursive_map, recursive_reduce]
parmapmethods = [parallel_map, recursive_reduce]
# names = []
sv = [5]
# for sv in [3,4,5,7, 10,12,15,20,21,25,28, 30,35,40,45,50,55,60,70, 80, 100, 120, 150, 200, 250,300, 350,400, 450, 500]:
for sv in [50,100,150]:
# for sv in [200,300,400]:
    for fn in fs:
        print('\n',fn, sv)
        if (fn, str(sv)) not in [(x[0], re.findall(r'[0-9]+', x[1])[0]) for x in  csdfs]:
            print('proceeding')
            csdfs.append((fn, 'par'+str(sv), try_do("highLevel/"+fn, parmethods, [sv] * 10)))
            print('par done')
            csdfs.append((fn, 'naive'+str(sv), try_do("highLevel/"+fn, naivemethods, [sv] * 10)))
            print('naive done')
            csdfs.append((fn, 'parreduce'+str(sv), try_do("highLevel/"+fn, parreducemethods, [sv] * 10)))
            print('parreduce done')
            csdfs.append((fn, 'parmap'+str(sv), try_do("highLevel/"+fn, parmapmethods, [sv] * 10)))
            print('parmap done')
#         w zip
#         csdfs.append((fn, 'par'+str(sv)+"_dz", try_do("highLevel/"+fn, [discrete_zip]+parmethods, [sv] * 10)))
#         csdfs.append((fn, 'naive'+str(sv)+"_dz", try_do("highLevel/"+fn, [discrete_zip]+naivemethods, [sv] * 10)))
#         csdfs.append((fn, 'parreduce'+str(sv)+"_dz", try_do("highLevel/"+fn, [discrete_zip]+parreducemethods, [sv] * 10)))
#         csdfs.append((fn, 'parmap'+str(sv)+"_dz", try_do("highLevel/"+fn, [discrete_zip]+parmapmethods, [sv] * 10)))

print('\n\n\n\n\n Complete')
csdfs = [x for x in csdfs if x[2] != False]
# printls(csdfs)
csdfs += [(x[0], x[1]+"_dz", discrete_zip(copy.deepcopy(x[2]['graph']), None)) for x in csdfs]
print('Discrete Zip complete')

print('\n\n\n\n\n Complete')
csdfs = [x for x in csdfs if x[2] != False]
# printls(csdfs)
extras = []
for x in csdfs:
    if 'dz' not in x[1]:
        print(x[0], x[1])
        csdf_copy = copy.deepcopy(x[2]['graph'])
        discrete_zip(csdf_copy, None)
        extras.append((x[0],  x[1]+"_dz", csdf_copy))
# csdfs += [(x[0], x[1]+"_dz", discrete_zip(copy.deepcopy(x[2]['graph']), None)) for x in csdfs]
csdfs += extras
print('Discrete Zip complete')
ddcsdfs = [(a[0], a[1], b) for a, b in dict([((x[0], x[1]), x[2]) for x in csdfs]).items()]



for csdf in csdfs:
    nbase =  'curr/'+csdf[0]+"_"+csdf[1]
    n1 =nbase+'_.xml'
    write_csdf(csdf[2]['graph'], n1)
    print("Written",n1)
    smushed = compose_maps(csdf[2]['graph'])
    n2 = nbase+'_cmap_.xml'
    write_csdf(smushed, n2)
    print("Written",n2)
    names += [n1,n2]
    
print('\n\n\n\n\n Finished')


# In[11]:


print('\n\n\n\n\n Complete')
csdfs = [x for x in csdfs if x[2] != False]
# printls(csdfs)
extras = []
for x in csdfs:
    if 'dz' not in x[1]:
        print(x[0], x[1])
        csdf_copy = copy.deepcopy(x[2]['graph'])
        discrete_zip(csdf_copy, None)
        extras.append((x[0],  x[1]+"_dz", csdf_copy))
# csdfs += [(x[0], x[1]+"_dz", discrete_zip(copy.deepcopy(x[2]['graph']), None)) for x in csdfs]
csdfs += extras
print('Discrete Zip complete')
ddcsdfs = [(a[0], a[1], b) for a, b in dict([((x[0], x[1]), x[2]) for x in csdfs]).items()]

for csdf in ddcsdfs:
    g = csdf[2] if 'dz' in csdf[1] else csdf[2]['graph']
    nbase =  'curr/'+csdf[0]+"_"+csdf[1]
    n1 =nbase+'_.xml'
    write_csdf(g, n1)
    print("Written",n1)
    smushed = compose_maps(g)
    n2 = nbase+'_cmap_.xml'
    write_csdf(smushed, n2)
    print("Written",n2)
    names += [n1,n2]


# In[51]:


for csdf in ddcsdfs:
    nbase =  'curr/'+csdf[0]+"_"+csdf[1]
    n2 = nbase+'_cmap_.xml'
    n1 =nbase+'_.xml'
    if n1 not in names and n2 not in names and 'mmTN' in nbase:
        print(n1, n2)
        g = csdf[2] if 'dz' in csdf[1] else csdf[2]['graph']
        write_csdf(g, n1)
        print("Written",n1)
        smushed = compose_maps(g)
        write_csdf(smushed, n2)
        print("Written",n2)
        names += [n1,n2]


# In[46]:


names


# In[ ]:





# In[41]:


ddcsdfs = [(a[0], a[1], b) for a, b in dict([((x[0], x[1]), x[2]) for x in csdfs]).items()]


# In[42]:


len(csdfs), len(ddcsdfs)


# In[45]:


len([x for x in ddcsdfs if 'dz' in x[1]])


# In[23]:


csdfs[354]


# In[ ]:


csdfs


# In[ ]:


csdfs = []

parmethods = [parallel_reduce, recursive_map, recursive_reduce]
recmethods = [recursive_map, recursive_reduce]

sv = [5]
for sv in [10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    for fn in fs:
        csdfs.append((fn, 'par'+str(sv), try_do("highLevel/"+fn, parmethods, [sv] * 10)))
        csdfs.append((fn, 'rec'+str(sv), try_do("highLevel/"+fn, recmethods, [sv] * 10)))

    csdfs = [x for x in csdfs if x[2] != False]
    # printls(csdfs)

    for csdf in csdfs:
        write_csdf(csdf[2]['graph'], 'csdf_xmls2/'+csdf[0]+"_"+csdf[1]+'_.xml')
        smushed = compose_maps(csdf[2]['graph'])
        write_csdf(smushed, 'csdf_xmls2/'+csdf[0]+"_"+csdf[1]+'_cmap_.xml')


# In[ ]:


n = p['graph']['nodes']
g = n[1].subfunc
printls(g['inputs'])
print(g['output'])


# In[ ]:


printgraph(g)


# In[ ]:


set(fs).difference(set(success_rec))


# In[ ]:


set(fs).difference(set(success_par))


# In[ ]:


success_rec


# In[ ]:





# okay so:
# 
# - managed to compile 18/21, both recursive and parallel
# - of the two that failed, is because the produce an Unknown datatype that ends up cycling back into a Map/Reduce so we don't have length data
# - but hey 85% success rate

# In[ ]:


ks = [c for c in csdfs if c[0] == 'mappy'and '4' in c[1]and '0' not in c[1]]
for k in ks:
    print(k[1])
    printgraph_simple(k[2]['graph'])
    print('\n')


# In[ ]:


from Generators import *
# k = try_do("highLevel/mappy", recmethods, [10] * 10)['graph']
# write_csdf(k, 'csdf_xmls/mappy.xml')
# printgraph_simple(k)


# In[ ]:



for i in [3,10,20,50,80,100,150,200,250,300,450,500]:
    for sv in [3,4,5,7, 10,12,15,20,21,25,28, 30,35,40,45,50,55,60,70, 80, 100, 120, 150, 200, 250,300, 350,400, 450, 500]:
        k = try_do("highLevel/userfun", recmethods, [sv] * 10)['graph']
        write_csdf(k, 'tstuf/uf_'+str(sv)+'_'+str(i)+'_rec_.xml', i)
        k = try_do("highLevel/userfun", parmethods, [sv] * 10)['graph']
        write_csdf(k, 'tstuf/uf_'+str(sv)+'_'+str(i)+'_par_.xml', i)


# In[ ]:



parmethods = [parallel_map, recursive_reduce]
recmethods = [recursive_map, recursive_reduce]
k = try_do("highLevel/asum", recmethods, [200] * 10)['graph']
write_csdf(k, 'parmapping/aaaamappy_rec_.xml')
k = try_do("highLevel/asum", parmethods, [200] * 10)['graph']
write_csdf(k, 'parmapping/aaaamappy_par_.xml')


# In[ ]:


from Generators import *


# In[ ]:


import timeit


parreducemethods = [parallel_reduce, recursive_map, recursive_reduce]
parmethods = [parallel_reduce, parallel_map, recursive_reduce]
naivemethods = [recursive_map, recursive_reduce]
parmapmethods = [parallel_map, recursive_reduce]
ls = range(1, 502, 50)
naive_ct = []
par_ct = []
for i in ls:
    naive_ct.append(timeit.timeit('try_do("highLevel/asum", naivemethods, [i])', number=100))
    par_ct.append(timeit.timeit('try_do("highLevel/asum", parmapmethods, [i])', number=100))


# In[ ]:





# In[ ]:


mypath = 'highlevel/'
fs = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.endswith('json')]
fs = [x for x in fs if 'mm' not in x] + ['mmNT']


# In[ ]:


parreducemethods = [parallel_reduce, recursive_map, recursive_reduce]
parmethods = [parallel_reduce, parallel_map, recursive_reduce]
naivemethods = [recursive_map, recursive_reduce]
parmapmethods = [parallel_map, recursive_reduce]


# In[ ]:



fs = ['mmNT']


names = []
csdfs = []


# In[ ]:


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
        nbase =  'curr/'+csdf[0]+"_"+csdf[1]
        n1 =nbase+'_.xml'
        n2 = nbase+'_cmap_.xml'
        write_csdf(csdf[2]['graph'], n1)
        smushed = compose_maps(csdf[2]['graph'])
        write_csdf(smushed, n2)

    print('Finished')

