#!/usr/bin/env python
# coding: utf-8

# * handle baby funcs and user funcs
# * sometimes v is a Value sometimes it's just given
# 
# 
# * VectorizeUserFunc? What?
# * wtf is up with Filter
# * Reduce vs ReduceSeq
# 
# Ops:
#     - Get = ._
#     - Value : val, type
#     VectorizeUserFun : function, arg
#     Transpose : arg
#     Join : arg
#     Zip : arg, arg
#     Map : function, arg
#     Filter : ?
#     ReduceSeq : ?
#     Reduce : function, val, arg
#     Lambda : fun(...) | \(...)
#     - Params
#     UserFuncs

# ./Debug/bin/kiter -f <file> -a PrintInfos

# * strip quotes from function?
# * what's the point of \?
# * handle getting el from tuple (Get)
# * :>> ??? = pipe in to next op I think
# * 0 vs 0.0f vs "0.0f" vs 1 vs 3.40282347e+38
# 
# Ops:
#     - Get = ._
#     - Value : val, type
#     VectorizeUserFun : function, arg
#     Transpose : arg
#     Join : arg
#     Zip : arg, arg
#     Map : function, arg
#     Filter : ?
#     ReduceSeq : ?
#     Reduce : function, val, arg
#     Lambda : fun(...) | \(...)
#     - Params
#     UserFuncs
# 
# Piping:
#     o : apply - f o g(x) = f(g(x))
#     \$ : piping  - f \$ x = f(x)
#     :>> : reverse piping - x :>> f = f(x)
# 
# Types:
#  'TupleType',
#  'Float4',
#  'Int',
#  'Float',
#  'ArrayType',

# In[1]:


import os, re, random
# from lxml import etree

def flatten(lol):
    return [val for sublist in lol for val in sublist]
def printls(ls):
    for item in ls:
        print(item)
        
def getcn(thingy):
    return thingy.__class__.__name__

program_names = []
programs = []
def load_programs():
    liftdir = os.getcwd()  + "/highLevel"
    global program_names
    program_names += os.listdir(liftdir)
    program_names = [x for x in program_names if 
                os.path.isfile(os.path.join(liftdir,x)) and ('.' not in x)]

    for program_name in program_names:
        with open(os.path.join(liftdir,program_name)) as f:
            programs.append(f.read())


# In[ ]:





# In[2]:


# Input: string
# Output: tuple: (string before brackets, 
#                 top-level contents of first brackets (or full string if no brackets),
#                 string after brackets)
# Error state: If brackets not closed
def inbc(s):
    if '(' not in s:
        return "",s,""
    start = s.index('(')
    open_ctr = 0
    for i in range(start+1, len(s)):
        if s[i] == '(':
            open_ctr += 1
        if s[i] == ')':
            if open_ctr == 0:
                return s[0:start], s[start+1:i], s[i+1:]
            else:
                open_ctr -= 1
    return False

def getcbc(s):
    if '{' not in s:
        return s
    start = s.index('{')
    open_ctr = 0
    for i in range(start+1, len(s)):
        if s[i] == '{':
            open_ctr += 1
        if s[i] == '}':
            if open_ctr == 0:
                return s[start+1:i]
            else:
                open_ctr -= 1
    return False

def getbc(s):
    return inbc(s)[1]

def inbc_rev(s):
    if '(' not in s:
        return "",s,""
    end = s.rindex(')')
    open_ctr = 0
    for i in range(end-1, -1, -1):
        if s[i] == ')':
            open_ctr += 1
        if s[i] == '(':
            if open_ctr == 0:
                return s[0:i], s[i+1:end], s[end+1:]
            else:
                open_ctr -= 1
    return False    

# Input: Overarching program
# Output: Main function
def getfunc(s):
    start = s.index('\nfun')
    s = getbc(s[start:])
    return re.sub(r"([a-zA-Z0-9]+)._([0-9]+)", r"Get(\g<1>, \g<2>)",s)

# Input: Overarching program
# Output: List of all SizeVar names
def getsvs(s):
    return re.findall(r'val\s+([a-zA-Z]+)\s*=\s*SizeVar\("[a-zA-Z]+"\)', s)

# Input: function
# Output: body of function
def getbody(s):
    i = s.index("=>")
    s = s[i+2:].strip()
    if s[0] == '(':
        s = getbc(s)
    if s[0] == '{':
        s = getcbc(s)
    if s[0] == '(':
        s = getbc(s)
    return s
def getmainbody(s):
    s = getfunc(s)
    return getbody(s)


# In[3]:


# Input: string of comma-separated items
# Output: list of top-level items
def commasplit(s):
    commas = [-1]
    open_ctr = 0
    for i in range(len(s)):
        if s[i] in ['(', '{', '[']:
            open_ctr += 1
        if s[i] in [')', '}', ']']:
            open_ctr -= 1
        if s[i] == ',' and open_ctr == 0:
            commas.append(i)
    commas.append(len(s))
    broken = [s[commas[j-1]+1:commas[j]] for j in range(1, len(commas))]
    return [x.strip() for x in broken if len(x.strip()) > 0]

# Input: string of comma-separated items
# Output: list of top-level items
def commasplit_rev(s):
    commas = [len(s)]
    open_ctr = 0
    for i in range(len(s)-1 , -1, -1):
        if s[i] in [')', '}', ']']:
            open_ctr += 1
        if s[i] in ['(', '{', '[']:
            open_ctr -= 1
        if s[i] == ',' and open_ctr == 0:
            commas.append(i)
    commas.append(-1)
    commas.reverse()
    broken = [s[commas[j-1]+1:commas[j]] for j in range(1, len(commas))]
    return [x.strip() for x in broken if len(x.strip()) > 0]

# Input: Main function with input types, input names, and function
# Output: list of pairs (input name, input type)
def getinputs(s):
    f = s.split("=>")[0]
    f = commasplit_rev(f)
    t = f[0:-1]
    n = getbc(f[-1])
    n = commasplit_rev(n)
    return [(a.strip(), b.strip()) for a, b in zip(n,t)]
def getmaininputs(s):
    return getinputs(getfunc(s))


# Input: lambda function
# Output: names of paramaters
def getnames(s):
    f = s.split("=>")[0]
    ns = getbc(f)
    return commasplit(ns)


# Input: Overarching program
# Output: Given subfunctions used in main function
def getsubfuncs(s):
    sfnames = re.findall(r"\ndef +([a-zA-Z0-9]+) *= *fun", s)
    return [(name, getbc(s[s.index(name):])) for name in sfnames]
def getsplit(s):
    return [x for x in re.split(r"[\(\){},\s]", s) if len(x) > 0]

def getNextOp(s):
    s = s.strip()
    return re.match(r"([A-Za-z0-9]+)", s).groups()[0]

def getNextParams(s):
    return commasplit_rev(getbc(s))
def hasparams(s):
    return '(' in s or '{' in s

def isfunction(s):
    if s.strip().startswith(("\(", "fun(")):
        return True
    return False

# Input: body of function
# Output: list of top-level functions separated by operators
def componentsplit(s):
    commas = [0]
    open_ctr = 0
    for i in range(len(s)):
        if s[i] in ['(', '{', '[']:
            open_ctr += 1
        if s[i] in [')', '}', ']']:
            open_ctr -= 1
        if open_ctr == 0:
            if s[i:].startswith("$"):
                commas += [i, i + 1]
            if s[i:].startswith(":>>"):
                commas += [i, i + 3]
            if re.match(r".\bo\b", s[i:]):
                commas += [i+1, i + 2]
    commas.append(len(s))
#     print(commas)
    broken = [s[commas[j-1]:commas[j]].strip() for j in range(1, len(commas))]
    return broken


# In[4]:


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
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='Value' value='"+self.value+"' kind='"+self.kind+"'>\n" + '\n'.join(map(str, self.output)) + "\n</actor>"

class Reduce(CSDFNode):
    def __init__(self, name, subfunc, initval, seq = False):
        self.name = name
        self.type =  "ReduceSeq" if seq else "Reduce"
        self.subfunc = subfunc
        self.initval = initval
        self.seq = seq
        self.input = Port('in', name + '_in', [1])
        self.output = Port('out', name + '_out', [1])
    def __str__(self):
        return "<actor name='"+self.name+"' type='"+self.type+"'>\n" + '\n'.join(map(str, [self.output, self.input])) + "\n</actor>"


class Map(CSDFNode):
    def __init__(self, name, subfunc):
        self.name = name
        self.subfunc = subfunc
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
    def __str__(self):
        return "<port type='"+self.direction+"' name='"+self.name+"' rate='"+','.join(map(str,self.rate))+"'/>"


# In[5]:


#    def __str__(self):
#         return "<port type='"+self.direction+"' name='"+self.name+"' rate='"+','.join(map(str,self.rate))+"'/>"


class SV:
    def __init__(self, name,  val):
        self.label = name
        self.val = val
    def __str__(self):
        return self.label + ": " + str(val)
class Float:
    def __init__(self):
        pass
    def size(self):
        return 4
    def __str__(self):
        return "Float"
class Int:
    def __init__(self):
        pass
    def size(self):
        return 1
    def __str__(self):
        return "Int"
class Array:
    def __init__(self, length, subdata):
        self.length = length
        self.subdata = subdata
    def size(self):
        return self.length * self.subdata.size()
    def __str__(self):
        return "Array (" + str(self.length) + ", " + str(self.subdata) + ")"
class Tuple:
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
class Unknown:
    def __init__(self):
        pass
    def size(self):
        return 40
    def __str__(self):
        return "Unknown"


# In[6]:


def unpack_program(program):
    node_ctr = 0
    def getname():
        nonlocal node_ctr
        node_ctr += 1
        return "n" + str(node_ctr)
    
    named_nodes = {}
    
    starting_inputs = getmaininputs(program)
    sizevars = getsvs(program)
    main_func = commasplit(getfunc(program))[-1]
    
    def mk_bby_func(bby, n_inputs):
        assert(len(set(bby.strip()).intersection(set("{}()[],. "))) == 0)
        if bby in ["mult", "add"]:
            new_node = Mather(getname(), bby)
            return {
                "nodes" : [new_node],
                "channels" : [],
                "inputs" : [new_node.new_inport(), new_node.new_inport()],
                "output" : new_node.output
            }
            
        new_node = UserFun(getname(), bby)
        for i in range(n_inputs):
            new_node.new_inport()
        return {
            "nodes" : [new_node],
            "channels" : [],
            "inputs" : new_node.input,
            "output" : new_node.output
        }
        
    
    # Output: {nodes : [Node], channels : [{Port, Port}], inputs : [Port], output : Port}
    def unpack_func(func):
        nodes = []
        channels = []
        myinputs = []
        local_names = getnames(func) if "=>" in func else []

        for name in local_names:
            new_node = Param(getname(), name)
            nodes.append(new_node)
            myinputs.append(new_node.input)
            assert(name not in named_nodes)
            named_nodes[name] = new_node  
        
        # Input: body
        # does: add nodes + channels to func list
        # Output: {in : Port, out :Port}
        def unpack_body(body):
            comps = componentsplit(body)
            assert(len(comps) > 0)
            assert(len(comps) % 2 == 1)
            
            curr = unpack_comp(comps[0])
            for i in range(1, len(comps), 2):
                assert(comps[i] in ["o", "$", ":>>"])
                new = unpack_comp(comps[i+1])
                if comps[i] == ":>>":
                    assert(new["in"] and curr["out"])
                    channels.append({"dst":new["in"], "src" : curr["out"]})
                    curr["out"] = new.get("out")
                else:
                    assert(curr["in"] and new["out"])
                    channels.append({"dst" : curr["in"], "src" : new["out"]})
                    curr["in"] = new.get("in")
            return curr
            
        
        # Input: single component
        # does: add nodes + channels to func list
        # Output: {in : Port, out :Port}
        def unpack_comp(comp):
            assert(len(componentsplit(comp)) == 1)
            op = getNextOp(comp)
            if op in named_nodes.keys():
                assert(op == comp.strip())
                src = named_nodes[op]
                src_port = src.new_outport()
                return({"out" : src_port})
            if op == "Value":
                args = getNextParams(comp)
                # It's the final node
                assert(len(args) == 2)
                assert(all([len(componentsplit(arg)) == 1 for arg in args]))
                new_node = Value(getname(), args[0], args[1])
                nodes.append(new_node)
                return({"out" : new_node.output})
            if op == "Get":
                args = getNextParams(comp)
                assert(len(args) == 2)
                assert(all([len(componentsplit(arg)) == 1 for arg in args]))
                new_node = Get(getname(), args[1])
                incoming = unpack_body(args[0])["out"]
                channels.append({"src" : incoming, "dst" : new_node.input})
                nodes.append(new_node)
                return({"out" : new_node.output})
            if op == "Transpose":
                assert(len(getNextParams(comp)) == 0)
                new_node = Transpose(getname())
                nodes.append(new_node)
                return({"in" : new_node.input, "out" : new_node.output})
            if op == "Join":
                assert(len(getNextParams(comp)) == 0)
                new_node = Join(getname())
                nodes.append(new_node)
                return({"in" : new_node.input, "out" : new_node.output})
            if op == "Zip":
                args = getNextParams(comp)
                new_node = Zip(getname())
                for arg in args:
                    new_in = new_node.new_inport()
                    new_out = unpack_body(arg)["out"]
                    channels.append({"src" : new_out, "dst" : new_in})
                nodes.append(new_node)
                return({"out" : new_node.output})
            if op in ["mult", "add"]:
                args = getNextParams(comp)
                new_node = Mather(getname(), op)
                for arg in args:
                    new_in = new_node.new_inport()
                    new_out = unpack_body(arg)["out"]
                    channels.append({"src" : new_out, "dst" : new_in})
                nodes.append(new_node)
                return({"out" : new_node.output})
            if op == "Map":
                args = getNextParams(comp)
                assert(len(args) == 1)
                f = args[0].strip()
                if f.startswith("VectorizeUserFun"):
                    f = getNextParams(f)[1]
                if isfunction(f):
                    child_func = unpack_func(getbc(f))
                elif '(' in f:
                    child_func = unpack_func(f)
                else:
                    child_func = mk_bby_func(f, 1)
                new_node = Map(getname(), child_func)
                nodes.append(new_node)
                return({"in" : new_node.input, "out" : new_node.output})
            if op in ["Reduce", "ReduceSeq"]:
                args = getNextParams(comp)
                assert(len(args) == 2)
                f = args[0].strip()
                if f.startswith("VectorizeUserFun"):
                    f = getNextParams(f)[1]
                if isfunction(f):
                    child_func = unpack_func(getbc(f))
                elif '(' in f:
                    child_func = unpack_func(f)
                else:
                    child_func = mk_bby_func(f, 2)
                init_val = args[1]
                new_node = Reduce(getname(), child_func, init_val, op == "ReduceSeq")
                nodes.append(new_node)
                return({"in" : new_node.input, "out" : new_node.output})
            else:
                new_node = UserFun(getname(), op)
                if hasparams(comp):
                    for arg in getNextParams(comp):
                        new_in = new_node.new_inport()
                        new_out = unpack_body(arg)["out"]
                        channels.append({"src" : new_out, "dst" : new_in})
                nodes.append(new_node)
                return({"out" : new_node.output})
        
        
        mybod = getbody(func) if "=>" in func else func
        res = unpack_body(mybod)
        if "=>" in func:
            assert(not res.get("in"))
        else:
            myinputs = [res["in"]]
        myoutput = res["out"]
        
        for name in local_names:
            named_nodes.pop(name)
        
        return {
            "nodes" : nodes,
            "channels" : get_channels(channels),
            "inputs" : myinputs,
            "output" : myoutput
        }
            
    parsed = unpack_func(main_func)
    return {
        "sizevars" : sizevars,
        "inputs" : starting_inputs,
        "code" : getfunc(program),
        "graph" : parsed
    }       


# In[7]:



def printgraph(g):
    n = g["nodes"]
    c = g["channels"]
    printls(n)
    printls(c)
    
    for node in n:
        if getcn(node).startswith(("Reduce", "Map", "Iterate")):
            print("\n       Subgraph of", getcn(node),"node ", node.name)
            printgraph(node.subfunc)


# In[8]:


def get_node_by_port(nodes, port_name):
    ns = [n for n in nodes if any([p.name == port_name for p in getports(n)])]
    assert(len(ns) == 1)
    return ns[0]

def get_connected_node(g, port_name):
    nodes = g['nodes']
    channels = g['channels']
    channel = [c for c in channels if c.src_port == port_name or c.dst_port == port_name]
    if len(channel) == 0:
        return None
    assert(len(channel) == 1)
    channel = channel[0]
    other_port_name = channel.dst_port if channel.src_port == port_name else channel.src_port 
    return get_node_by_port(nodes, other_port_name)


# In[9]:


def getin(n):
    if type(n.input) is list:
        return n.input
    else:
        return [n.input]
def getout(n):
    if type(n.output) is list:
        return n.output
    else:
        return [n.output]

def check_correct(g):
    ns = g["nodes"]
    cs = g["channels"]
    nd = dict([(n.name, n) for n in ns])
    
    for c in cs:
        assert(c.dst_port in [p.name for p in getin(nd[c.dst_act])])
        assert(c.src_port in [p.name for p in getout(nd[c.src_act])])


# In[10]:


# TODO: remove and fix actual cascade
def add_dt_to_channels(program):
    g = program['graph']
    cs = g["channels"]
    nd = dict([(n.name, n) for n in g["nodes"]])
    for c in cs:
        in_node = nd[c.src_act]
        in_dt = in_node.get_dt()
        c.add_dt(in_dt)
    return program


# In[11]:


def smush_rede(g):
    ns = g["nodes"]
    cs = g["channels"]
    n_dict = dict([(n.name, n) for n in ns])
    rds = [ c for c in cs if ((getcn(n_dict[c.src_act]) == "Dearray" 
                        and getcn(n_dict[c.dst_act]) == "Rearray")
                       or (getcn(n_dict[c.src_act]) == "Rearray" 
                        and getcn(n_dict[c.dst_act]) == "Dearray"))]
    for rd in rds:
        src = n_dict[rd.src_act]
        dst = n_dict[rd.dst_act]
        assert(getcn(src) in ["Rearray", "Dearray"])
        assert(getcn(dst) in ["Rearray", "Dearray"])
        if src.output.rate == dst.input.rate:
            inc_c = [c for c in cs if  c.dst_port == src.input.name][0]
            out_c = [c for c in cs if c.src_port == dst.output.name][0]
            assert(inc_c.datatype == out_c.datatype)
            new_c = Channel(inc_c.src_act, out_c.dst_act, inc_c.src_port, out_c.dst_port)
            new_c.add_dt(inc_c.datatype)
            cs = [c for c in cs if c not in [inc_c, out_c, rd]] + [new_c]
            ns = [n for n in ns if n not in [src, dst]]
    g["nodes"] = ns
    g["channels"] = cs
    
    return g


# In[12]:


xml_hdr = """<?xml version="1.0" encoding="UTF-8"?>
<sdf3 type="csdf" version="1.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://www.es.ele.tue.nl/sdf3/xsd/sdf3-csdf.xsd">
<applicationGraph name='autogen'>
    <csdf name='autogen' type='autogen'>
    """

xml_mid = """
    </csdf>

    <csdfProperties>
    """

xml_end = """
    </csdfProperties>

</applicationGraph>

</sdf3>

"""

def check_and_get_phasecount(node):
    ports = getports(node)
    phase_counts = [len(p.rate) for p in ports]
    assert(len(set(phase_counts)) == 1)
    return phase_counts[0]

def get_actor_properties(node):
    ext = 4
    if getcn(node) in ["Param", "Value", "Get"]:
        ext = 1
    if getcn(node) in ["Split", "Join", "Dearray", "Rearray", "Zip"]:
        ext = 3
    if getcn(node) in ["Transpose", "UserFun"]:
        ext = 10
    phase_count = str(ext)+(',' + str(ext)) * (check_and_get_phasecount(node) - 1)
    properties = """<actorProperties actor='{0}'>
        <processor type='cluster_0' default='true'>
            <executionTime time='{1}'/>
        </processor>
    </actorProperties>
    """.format(node.name, phase_count)
    return properties

def get_sdf(g):
    nodes = g["nodes"]
    channels = g["channels"]
    
    # Build
    sdf = xml_hdr
    sdf += "\n".join([str(node) for node in nodes])
    sdf += "\n".join([str(channel) for channel in channels])
    sdf += xml_mid
    sdf += "\n".join([get_actor_properties(node) for node in nodes])
    sdf += xml_end
    
    return sdf
    
    
def get_sdf_from_hl(p):
    return get_sdf(smush_rede(explode(unpack_program(p)["graph"])))


# In[13]:


def explode(graph):
    def_len = 10
    nodes = graph["nodes"]
    channels = graph["channels"]
    
    # get Map nodes
    mapnodes = [n for n in nodes if getcn(n) == "Map"]
    # explode Map nodes
    for node in mapnodes:
        name = node.name
        da_in = node.input
        da_out = Port("out", name + "XDA_out", [def_len])
        da = Dearray(name + "XDA", da_in, da_out)
        ra_out = node.output
        ra_in = Port("in", name + "XRA_in", [def_len])
        ra = Rearray(name + "XRA", ra_in, ra_out, "Map")
        
        subgraph = explode(node.subfunc)
        nodes += subgraph["nodes"]
        nodes += [da, ra]
        
        old_in_channel = [c for c in channels if c.dst_act == name]
        old_out_channel = [c for c in channels if c.src_act == name]
        assert(len(old_in_channel) <= 1)
        assert(len(old_out_channel) <= 1)
        if len(old_in_channel) == 1:
            old_in_channel[0].dst_act = da.name
        if len(old_out_channel) == 1:
            old_out_channel[0].src_act = ra.name
        channels += subgraph["channels"]
        so = subgraph["output"].name
        si = subgraph["inputs"][0].name
        channels += [Channel(da.name, 
                             get_node_by_port(subgraph["nodes"], si).name, 
                             da_out.name, si), 
                     Channel(get_node_by_port(subgraph["nodes"], so).name, 
                             ra.name, so, ra_in.name)]
    # remove Map nodes
    nodes = [n for n in nodes if getcn(n) != "Map"]
    
    
    # get Reduce nodes
    reducenodes = [n for n in nodes if getcn(n).startswith("Reduce")]
    # explode Reduce nodes
    for node in reducenodes:
        name = node.name
        da_in = node.input
        da_out = Port("out", name + "XDA_out", [def_len])
        da = Dearray(name + "XDA", da_in, da_out)
        ra_in = Port("in", name + "XRA_in", [1])
        ra_out = node.output
        ra = Rearray(name + "XRA", ra_in, ra_out, "Reduce")
        
        # make param for recursion
        para = Param(name + "XPARA", "recursion")
        para.input.rate = [1] * def_len
        para_out_0 = para.new_outport([1] * def_len)
        para_out_1 = para.new_outport([0] * (def_len - 1) + [1])
        
        # add nodes
        subgraph = explode(node.subfunc)
        nodes += subgraph["nodes"]
        nodes += [da, ra, para]
        
        # add channels
        old_in_channel = [c for c in channels if c.dst_act == name]
        old_out_channel = [c for c in channels if c.src_act == name]
#         printls(old_in_channel)
#         printls(old_out_channel)
        if len(old_in_channel) == 1:
            old_in_channel[0].dst_act = da.name
        if len(old_out_channel) == 1:
            old_out_channel[0].src_act = ra.name
        
        si = subgraph["inputs"]
        so = subgraph["output"].name
        channels += [Channel(para.name, get_node_by_port(subgraph["nodes"], si[1].name).name, 
                             para_out_0.name, si[1].name, 1),
                     Channel(da.name, get_node_by_port(subgraph["nodes"], si[0].name).name, 
                             da_out.name, si[0].name),
                     Channel(get_node_by_port(subgraph["nodes"], so).name, para.name,
                             so, para.input.name),
                     Channel(para.name, ra.name,
                             para_out_1.name, ra_in.name)]
        channels += subgraph["channels"]
        
    # remove Reduce nodes
    nodes = [n for n in nodes if not getcn(n).startswith("Reduce")]
    
    
    graph["nodes"] = nodes
    graph["channels"] = channels
    return graph
            


# In[14]:


def batch_read_and_write(source_dir, target_dir):
    program_names = os.listdir(source_dir)
    program_names = [x for x in program_names if 
                os.path.isfile(os.path.join(source_dir, x)) and ('.' not in x)]
    
    for program_name in program_names:
        with open(os.path.join(source_dir,program_name)) as f:
            program = f.read()
        sdf = get_sdf_from_hl(program)
        with open(os.path.join(target_dir,program_name+".txt"), "w") as f:
            f.write(sdf)


# In[15]:



def printgraph_simple(g):
    n = g["nodes"]
    c = g["channels"]
    printls([node.name + ": " + getcn(node) for node in n])
    printls([(channel.src_act, channel.dst_act)] for channel in c)
    
    for node in n:
        if getcn(node).startswith(("Reduce", "Map", "Iterate")):
            print("\n       Subgraph of", getcn(node),"node ", node.name)
            printgraph_simple(node.subfunc)


# In[16]:


def get_output_node(g):
    return [n for n in g['nodes'] if n.output == g['output']][0]

def get_input_nodes(g):
    return [n for n in g['nodes'] if n.input in g['inputs']]    
    

def magic_sv(i):
    myopts = [10,20,30,50,5,4]
    return random.choice(myopts)

    
def channel_is_cyclic(src, dst, g):
    ns = g["nodes"]
    cs = g["channels"]
    nd = dict([(n.name, n) for n in ns])
    cbsp = dict([(c.src_port, c) for c in cs])
    
    
    visited = []
    
    def check_branch(node):
        if node == get_output_node(g):
            return False
        if node in visited:
            return False
        if node.name == src.name:
            return True
        visited.append(node)
        out_channels = [cbsp[p.name] for p in getout(node)]
        branches = [nd[x.dst_act] for x in out_channels]
        return any([check_branch(x) for x in branches])
    
    return check_branch(dst)
        
        
    
    
    
# TODO: Refactor so that datatype is a function of channel, not node
# TODO: is there an okay way to rollback info gained after an Unknown?
# TODO: This is a godawful mess please fix
# I mean like duh
# NOTE! Must be done before smush_rede.
def cascade(program, sv_maker = magic_sv):
    g = program["graph"]
    ns = g["nodes"]
    cs = g["channels"]
    nd = dict([(n.name, n) for n in ns])
    cbsp = dict([(c.src_port, c) for c in cs])
    svs = dict([(program['sizevars'][i], sv_maker(i)) for i in range(len(program['sizevars']))])
    
    visited = []
    
    def form_type(s):
        op = getNextOp(s).lower()
        if op.startswith('float'):
            return Float()
        elif op.startswith('int'):
            return Int()
        elif op.startswith('array'):
            paras = getNextParams(s)
            assert(len(paras) == 2)
            sd = form_type(paras[0])
            try:
                l = int(paras[1].strip())
            except:
                l = svs[paras[1]] if 'SizeVar' not in paras[1] else magic_sv()
            return Array(l, sd)
        elif op.startswith('tuple'):
            paras = getNextParams(s)
            t = Tuple()
            [t.add_subdata(form_type(p)) for p in paras]
            return t
        else: 
            return Unknown()
    
    in_types = [(k, form_type(x) )for k, x in program["inputs"]]
    
    
    dearray_by = {}
    def def_odt(node, inc_dt):
        try:
            cn = getcn(node)
            if cn == "Param":
                if node.label == 'recursion':
                    node_num = re.findall(r'[0-9]+', node.name)[0]
                    rep_vec = dearray_by[node_num]
                    node.input.rate = [1] * rep_vec
                    node.output[0].rate = [1] * rep_vec
                    node.output[1].rate = para_out_1 = [0] * (rep_vec - 1) + [1]
                return inc_dt # e -> e
            elif cn == "Get":
                return inc_dt.subdatals[int(node.idx)]  #T(e, e, e, e...) -> e
            elif cn == "Transpose":
                sup_len = inc_dt.length
                sub_arr = inc_dt.subdata
                sub_len = sub_arr.length
                return Array(sub_len, Array(sup_len, sub_arr.subdata))
                # A( A(e, M), N ) -> A( A(e, N), M ) 
            elif cn == "Dearray":
                node_num = re.findall(r'[0-9]+', node.name)[0]
                node.output.rate = [inc_dt.length]
                dearray_by[node_num] = inc_dt.length
                print("DEARRAY ", node_num)
                return inc_dt.subdata # A(e, N) -> e
            elif cn == "Rearray":
                node_num = re.findall(r'[0-9]+', node.name)[0]
                new_len = 1 if node.masternode == "Reduce" else dearray_by[node_num]
                node.input.rate = [new_len]
                return Array(new_len, inc_dt) # e -> A(e, ?)
            elif cn == "Join":
                sup_len = inc_dt.length
                sub_arr = inc_dt.subdata
                sub_len = sub_arr.length
                return Array(sub_len * sup_len,  sub_arr.subdata)
                # A( A(e, N), M) -> A(e, N*M)
            elif cn == "Value":
                return inc_dt
            elif cn == "Zip":
                lengths = [x.length for x in inc_dt]
                subdatas = [x.subdata for x in inc_dt]
                assert(len(set(lengths)) == 1)
                return Array(lengths[0], Tuple(subdatas))
                # A(e), A(r) -> A(T(e, r))
            elif cn == "Mather":
                return inc_dt # e, e -> e    
            elif cn == "UserFun":
                return Unknown() # e -> b
        except AttributeError as e:
            print("ERROR ERROR ERROR")
            if e.args[0] == "'Unknown' object has no attribute 'length'":
                return Unknown()
            else:
                raise
        except KeyError:
            default = 10
            if cn == "Param":
                node.input.rate = [1] * default
                node.output[0].rate = [1] * default
                node.output[1].rate = para_out_1 = [0] * (default - 1) + [1]
                return inc_dt # e -> e
            elif cn == "Rearray":
                node.input.rate = [default]
                return Array(default, inc_dt) # e -> A(e, ?)
            
    
    
    
    def push(node):
        if node.name in visited or node == get_output_node(g):
            return
        visited.append(node.name)
        assert(node.has_dt())
        my_dt = node.get_dt()
        print("Push ", node.name, getcn(node))
        
        output_nodes = [get_connected_node(g, out.name) for out in getout(node)]
        output_ports = getout(node)
        output_channels = [cbsp[p.name] for p in output_ports]
        output_nodes2 = [nd[c.dst_act] for c in output_channels]
        assert(output_nodes == output_nodes2)
        print([n.name for n in output_nodes])
        
        for out_node in output_nodes:
            print(out_node.name)
            inc_nodes = [get_connected_node(g, inc.name) for inc in getin(out_node)]
            if not all([x.has_dt() or channel_is_cyclic(x, out_node, g)
                        for x in inc_nodes] ):
                print("          ", out_node.name, " not all")
                continue
            if getcn(out_node) != 'Zip':
                new_dt = def_odt(out_node, my_dt)
                out_node.add_dt(new_dt)
                push(out_node)
            else:
                my_dts = [get_connected_node(g, inc.name).datatype for inc in getin(out_node)]
                new_dt = def_odt(out_node, my_dts)
                out_node.add_dt(new_dt)
                push(out_node)
    
    
    # set types of input paramaters:
    for k, dt in in_types:
        param_nodes = [n for n in ns if getcn(n) == 'Param' and n.label == k]
        assert(len(param_nodes) == 1)
        param_node = param_nodes[0]
        param_node.add_dt(dt)
        push(param_node)
    
    return program


# In[17]:


def rollout(leng, tgtdir):
    mmnn = programs[1]
    p = unpack_program(mmnn)
    g = p['graph']
    explode(g)
    cascade(p, lambda x: leng)
    add_dt_to_channels(p)
    smush_rede(g)
    check_correct(g)
    sdf = get_sdf(g)
    with open(tgtdir + "mmnn_" + str(leng) + ".xml", "w") as f:
        f.write(sdf)
        
[rollout(x, "/Users/hebe/Dropbox/Yale-NUS/Yr4/capstone/kiter/mmnn/") for x in range(5,1000,50)]


# In[ ]:


list(range(1,100,5))


# In[ ]:



source_dir = "/Users/hebe/Dropbox/Yale-NUS/Yr4/capstone/forrealsies/lift/highLevel"
target_dir = "/Users/hebe/Dropbox/Yale-NUS/Yr4/capstone/helpers/sdfxmls"
batch_read_and_write(source_dir, target_dir)


# In[ ]:


load_programs()

parsed_programs =[unpack_program(p) for p in programs]
for p in parsed_programs:
    ex = explode(p["graph"])
    cascade(p)
    add_dt_to_channels(p)
    ex = smush_rede(ex)
    check_correct(ex)
    p["exploded"] = ex
sdfs = [get_sdf_from_hl(p) for p in programs]


# In[ ]:


for i in range(len(programs)):
    print(i)
    p = programs[i]
    h = unpack_program(p)
    explode(h['graph'])
    cascade(h)
    add_dt_to_channels(h)


# In[ ]:


pp = parsed_programs[0]
gg = pp["exploded"]
nn = gg["nodes"]
cc = gg["channels"]
nd = dict([(n.name, n) for n in nn])


# In[ ]:


program_names


# In[ ]:





# In[ ]:


h = unpack_program(programs[18])
explode(h['graph'])
cascade(h)

add_dt_to_channels(h)


# In[ ]:


for i in range(len(programs)):
    print(i)
    h = unpack_program(programs[i])
    explode(h['graph'])
    cascade(h)

    add_dt_to_channels(h)


# In[ ]:


g = h['graph']
cc = g['nodes']
for i in range(len(cc)):
    print(i, cc[i].name)


# In[ ]:


channel_is_cyclic(cc[3], cc[13] , h)


# In[ ]:





# In[ ]:


h = unpack_program(programs[18])
explode(h['graph'])
printgraph(h['graph'])


# In[ ]:


print(h['code'])


# In[ ]:



printgraph_simple(h['graph'])


# In[ ]:


mmnn = unpack_program(programs[1])
mmnn['code']


# In[ ]:


float("2.296253762e-10")


# In[ ]:


a=[getmaininputs(k) for k in programs][0]


# * keep stack of dearray lens
# * on rearray 1) check master node 2) if map, pop dearray len 3) if reduce, pop but use len 1
# 
