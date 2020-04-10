#!/usr/bin/env python
# coding: utf-8

# In[244]:


liftdir = "/Users/hebe/Dropbox/Yale-NUS/Yr4/capstone/forrealsies/lift"
object_name = "AllPrograms"

opening = """package ir.printer

import ir._
import ir.ast._
import lift.arithmetic._
import opencl.ir._
import opencl.ir.pattern._
import opencl.ir.abs

object """ + object_name + "{\n"


# In[245]:


import os, re

def flatten(lol):
    return [val for sublist in lol for val in sublist]


# In[246]:


programs = os.listdir(liftdir + "/highLevel")
programs = [x for x in programs if 
            os.path.isfile(os.path.join(liftdir,"highLevel",x)) and ('.' not in x)]


# In[247]:


parts = []

for program in programs:
    with open(os.path.join(liftdir,"highLevel",program)) as f:
        p = f.read()
    parts += p.split('\n\n')


# In[248]:


sizevars = [x for x in parts if "SizeVar" in x and x.startswith("val")]
arraytypes = set([x for x in parts if "ArrayType" in x and x.startswith("val")])
userfuns = set([x for x in parts if "UserFun" in x and x.startswith("val")])
helpers = set([x for x in parts if x.startswith("def")])
mains  = [x for x in parts if x.startswith("fun")]
others = [x for x in parts if 
          len(x) > 5 and
          x not in list(sizevars) + list(helpers) + list(mains) + list(userfuns) + list(arraytypes)]


# In[249]:


for other in others:
    print(other + "\n\n")


# In[250]:



print(set(re.findall(r"SizeVar\(\"([A-Z])\"\)", "".join(sizevars))))
print(set(re.findall(r"val\s+([A-Za-z]+)\s+=\s+SizeVar", "".join(sizevars))))


# In[251]:


sv = flatten([x.split('\n') for x in sizevars])
sv = [x.split(')')[0] + ')' for x in sv if 'SizeVar' in x]
sizevars = set(sv)
set(sv)


# In[252]:


helper_names = re.findall(r"def ([a-zA-Z0-9]+)", ''.join(helpers))
userfun_names = re.findall(r"val\s+([a-zA-Z0-9]+)\s+=\s+UserFun", ''.join(userfuns))

def check_by_name(name):
    return [x for x in parts if name in x[0:10+len(name)]]

duplicates = []
print("Checks:")
if len(set(helper_names)) != len(helpers):
    print("WARNING: duplicate helper names")
    duplicates +=[x for x in helper_names if helper_names.count(x) > 1]
else:
    print("helpers are a-go")
    
if len(set(userfun_names)) != len(userfuns):
    print("WARNING: duplicate userfun names")
    duplicates +=[x for x in userfun_names if userfun_names.count(x) > 1]
else:
    print("userfuns are a-go")

if len(mains) != len(programs):
    print("WARNING: Incorrect number of main programs")
else:
    print("Correct number of main programs")
    
for dupl in set(duplicates):
    print('\n\n'+dupl)
    for version in check_by_name(dupl):
        print('\n'+version)


# In[253]:


mains = ['val ' + x[0] + ' = ' + x[1] for x in zip(programs, mains)]

full_set = (opening + "\n" + "\n".join(sizevars)
             + "\n" + "\n".join(arraytypes)
             + "\n" + "\n".join(userfuns)
             + "\n" + "\n".join(helpers)
             + "\n" + "\n".join(mains)
             + "\n}"
           )


# In[254]:


with open(liftdir + "/src/main/ir/printer/" + object_name + ".scala", 'w') as f:
    f.write(full_set)
    
print("The available programs are:\n","\n".join(programs))


# In[ ]:





# In[239]:


with open(liftdir + "/highLevel/mriqPhiMag") as f:
    pp = f.read()


# In[241]:


a = pp.split('\n\n')


# In[243]:


len(a)


# In[256]:


print("\n".join(['TestPrinter("./printed/", "'+x+'_v1", '+x+')' for x in programs]))


# In[ ]:




