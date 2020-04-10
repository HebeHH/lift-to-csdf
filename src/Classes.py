#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


# IR CLASSES


class Channel:
    def __init__ (self, src_act, dst_act,  src, dst, init_token = 0):
        self.name = src+"_to_"+dst
        self.src_port = src
        self.src_act = src_act
        self.dst_port = dst
        self.dst_act = dst_act
        self.init_token = init_token
        self.size = 1
        self.datatype = False
    def get_size(self):
        if self.datatype:
            return self.datatype.size()
        return self.size
    def __str__(self):
        return "<channel name='"+self.name+"' srcActor='"+self.src_act+"' srcPort='"+self.src_port+"' dstActor='"+self.dst_act+"' dstPort='"+self.dst_port+"' size='"+str(self.get_size())+ "' dtype='"+str(self.datatype)+ "' initialTokens='"+str(self.init_token)+"'/>"
    def add_dt(self, dt):
        self.datatype = dt
    def get_dt(self):
        return self.datatype

class Port:
    def __init__ (self, direction, name, rate):
        self.name = name
        self.direction = direction
        self.rate = rate
    def __str__(self):
        return "<port type='"+self.direction+"' name='"+self.name+"' rate='"+','.join(map(str,self.rate))+"'/>"

def get_channel(src, dst):
    return Channel(src.name.split("_")[0], dst.name.split("_")[0], src.name, dst.name)
# INPUT: list of dicts of ports - [{'dst' : Port, 'src' : Port}]
# OUTPUT: list of channels - [Channel]
def get_channels(ls):
    return [get_channel(x["src"], x["dst"]) for x in ls]

class CSDFNode:
    def __init__():
        pass
    def add_dt(self, dt):
        self.datatype = dt
    def get_dt(self):
        return self.datatype
    def has_dt(self):
        try:
            if self.datatype == True:
                return True
            else:
                return True
        except:
            return False
        
class Param(CSDFNode):
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.input = Port('in', name + '_in', [1])
        self.out_ctr = 0
        self.output = []
    def new_outport(self, r = [1]):
        np = Port('out', self.name + "_x"+str(self.out_ctr)+ '_out', r)
        self.output.append(np)
        self.out_ctr += 1
        return np
    def __str__(self):
        return "<actor name='"+self.name+"' type='Param' >\n" + '\n'.join(map(str, [self.input]+self.output)) + "\n</actor>"

class Get(CSDFNode):
    def __init__(self, name, idx):
        self.name = name
        self.input = Port('in', name + '_in', [1])
        self.output = Port('out', name + '_out', [1])
        self.idx = idx
    def __str__(self):
        return "<actor name='"+self.name+"' type='Get' >\n" + '\n'.join(map(str, [self.input, self.output])) + "\n</actor>"

class Transpose(CSDFNode):
    def __init__(self, name):
        self.name = name
        self.input = Port('in', name + '_in', [1])
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='Transpose' >\n" + '\n'.join(map(str, [self.input, self.output])) + "\n</actor>"

    
class Dearray(CSDFNode):
    def __init__(self, name, portin, portout):
        self.name = name
        self.input = portin
        self.output = portout
    def __str__(self):
        return "<actor name='"+self.name+"' type='Dearray' >\n" + '\n'.join(map(str, [self.input, self.output])) + "\n</actor>"
    

        
class Rearray(CSDFNode):
    def __init__(self, name, portin, portout, mstr):
        self.name = name
        self.input = portin
        self.output = portout
        self.masternode = mstr
    def __str__(self):
        return "<actor name='"+self.name+"' type='Rearray' >\n" + '\n'.join(map(str, [self.input, self.output])) + "\n</actor>"
    
class Splitter(CSDFNode):
    def __init__(self, name, routes, groupsize, remaining):
        self.name = name
        self.input = Port('in', name + '_in', [groupsize]*(routes))
        self.output = [Port('out', name + '_out_x' +str(i), [0]*(i)+[groupsize]+[0]*(routes - i - 1)) 
                       for i in range(routes)]
        if remaining > 0:
            self.output.append(Port('out', name + '_out_x'+str(routes), [0]*(routes)+[remaining]))
    def __str__(self):
        return "<actor name='"+self.name+"' type='Splitter' >\n" + '\n'.join(map(str, [self.input]+ self.output)) + "\n</actor>"
    
class Joiner(CSDFNode):
    def __init__(self, name, routes, groupsize):
        self.name = name
        self.output = Port('out', name + '_out', [routes] * groupsize)
        self.input = [Port('in', name + '_in_x' +str(i), [1] * groupsize) 
                       for i in range(routes)]
    def __str__(self):
        return "<actor name='"+self.name+"' type='Joiner' >\n" + '\n'.join(map(str, self.input+ [self.output])) + "\n</actor>"
    
    
class Join(CSDFNode):
    def __init__(self, name):
        self.name = name
        self.input = Port('in', name + '_in', [1])
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='Join' >\n" + '\n'.join(map(str, [self.input, self.output])) + "\n</actor>"
    
# Must have new value for every attachment
class Value(CSDFNode):
    def __init__(self, name, value, kind):
        self.name = name
        self.value = value
        self.kind = kind
        self.input = []
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='Value' value='"+str(self.value)+"' kind='"+str(self.kind)+"'>\n" + str(self.output) + "\n</actor>"

class Reduce(CSDFNode):
    def __init__(self, name, subfunc, initval, seq = False):
        self.name = name
        self.type =  "ReduceSeq" if seq else "Reduce"
        self.subfunc = subfunc
        self.initval = initval
        self.seq = seq
        self.rep = 1
        self.input = Port('in', name + '_in', [1])
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='"+self.type+"'>\n" + '\n'.join(map(str, [self.output, self.input])) + "\n</actor>"


class Map(CSDFNode):
    def __init__(self, name, subfunc):
        self.name = name
        self.subfunc = subfunc
        self.rep = 1
        self.input = Port('in', name + '_in', [1])
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='Map'>\n" + '\n'.join(map(str, [self.output, self.input])) + "\n</actor>"
    
class Zip(CSDFNode):
    def __init__(self, name : str):
        self.name = name
        self.input = []
        self.in_ctr = 0
        self.output = Port('out', name + '_out', [1])
    def new_inport(self):
        np = Port('in', self.name + "_x"+str(self.in_ctr)+ '_in', [1])
        self.input.append(np)
        self.in_ctr += 1
        return np
    def __str__(self):
        return "<actor name='"+self.name+"' type='Zip'>\n" + '\n'.join(map(str, [self.output] + self.input)) + "\n</actor>"

class Zippee(CSDFNode):
    def __init__(self, name : str):
        self.name = name
        self.input = []
        self.in_ctr = 0
        self.output = Port('out', name + '_out', [1])
    def new_inport(self):
        np = Port('in', self.name + "_x"+str(self.in_ctr)+ '_in', [1])
        self.input.append(np)
        self.in_ctr += 1
        return np
    def __str__(self):
        return "<actor name='"+self.name+"' type='Zippee'>\n" + '\n'.join(map(str, [self.output] + self.input)) + "\n</actor>"


class Mather(CSDFNode):
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
        self.input = []
        self.in_ctr = 0
        self.output = Port('out', name + '_out', [1])
    def new_inport(self):
        np = Port('in', self.name + "_x"+str(self.in_ctr)+ '_in', [1])
        self.input.append(np)
        self.in_ctr += 1
        return np
    def __str__(self):
        return "<actor name='"+self.name+"' type='"+self.kind+"'>\n" + '\n'.join(map(str, [self.output] + self.input)) + "\n</actor>"

class UserFun(CSDFNode):
    def __init__(self, name : str, label):
        self.name = name
        self.label = label
        self.input = []
        self.in_ctr = 0
        self.output = Port('out', name + '_out', [1])
    def new_inport(self):
        np = Port('in', self.name + "_x"+str(self.in_ctr)+ '_in', [1])
        self.input.append(np)
        self.in_ctr += 1
        return np
    def __str__(self):
        return "<actor name='"+self.name+"' type='UserFun'  label='"+self.label+"'>\n" + '\n'.join(map(str, [self.output] + self.input)) + "\n</actor>"

def getports(node):
    return (node.input if type(node.input) == list else [node.input]) + (node.output if type(node.output) == list else [node.output])
    

# In[3]:


# TYPE CLASSES
class TYPECLASS:
    pass
class SV(TYPECLASS):
    def __init__(self, name,  val):
        self.label = name
        self.val = val
    def __str__(self):
        return self.label + ": " + str(val)
class Float(TYPECLASS):
    def __init__(self):
        pass
    def size(self):
        return 4
    def __str__(self):
        return "Float"
class Int(TYPECLASS):
    def __init__(self):
        pass
    def size(self):
        return 1
    def __str__(self):
        return "Int"
class Array(TYPECLASS):
    def __init__(self, length, subdata):
        self.length = length
        self.subdata = subdata
    def size(self):
        return self.length * self.subdata.size()
    def __str__(self):
        return "Array (" + str(self.length) + ", " + str(self.subdata) + ")"
class Tuple(TYPECLASS):
    def __init__(self, subdatals = []):
        self.subdatals = subdatals
    def length():
        return len(self.subdatals)
    def add_subdata(self, subdata):
        self.subdatals.append( subdata)
    def size(self):
        return sum([sd.size() for sd in self.subdatals])
    def __str__(self):
        return "Tuple (" + ", ".join([str(x) for x in self.subdatals] )+ ")"
class Unknown(TYPECLASS):
    def __init__(self):
        pass
    def size(self):
        return 40
    def __str__(self):
        return "Unknown"


# In[ ]:




from HelperFuncs import *

