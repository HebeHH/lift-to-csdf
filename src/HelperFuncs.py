#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, re, random
# In[7]:


# PARSER HELPER FUNCTIONS

def flatten(lol):
    return [val for sublist in lol for val in sublist]
def printls(ls):
    for item in ls:
        print(item)
        
def getcn(thingy):
    return thingy.__class__.__name__

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
#     print("hello")
#     print(s)
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



# In[3]:


# Graph Printers


def printgraph(g):
    n = g["nodes"]
    c = g["channels"]
    printls(n)
    printls(c)
    
    for node in n:
        if getcn(node).startswith(("Reduce", "Map", "Iterate")):
            print("\n       Subgraph of", getcn(node),"node ", node.name)
            printgraph(node.subfunc)
            

def printgraph_simple(g, dt = False):
    n = g["nodes"]
    c = g["channels"]
    printls([node.name + ": " + getcn(node) + (": "+str(node.datatype) if dt else "") for node in n])
    printls([(channel.src_act, channel.dst_act, str(channel.datatype))] for channel in c)
    
    for node in n:
        if getcn(node).startswith(("Reduce", "Map", "Iterate")):
            print("\n       Subgraph of", getcn(node),"node ", node.name)
            printgraph_simple(node.subfunc, dt)


# In[4]:


# Graph helper functions

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
    
def get_output_node(g):
    return [n for n in g['nodes'] if n.output == g['output']][0]

def get_input_nodes(g):
    return [n for n in g['nodes'] if n.input in g['inputs']]  
    
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
        

def check_correct(g):
    ns = g["nodes"]
    cs = g["channels"]
    nd = dict([(n.name, n) for n in ns])
    
    for c in cs:
        assert(c.dst_port in [p.name for p in getin(nd[c.dst_act])])
        assert(c.src_port in [p.name for p in getout(nd[c.src_act])])


# In[5]:



    
    
# CSDF XML Generator


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
#     print(node.name)
#     print(phase_counts)
    assert(len(set(phase_counts)) == 1)
    return phase_counts[0]

def get_actor_properties(node, i):
    execution_times = {
        "Param":10, 
        "Value":0, 
        "Get":2,
        "Splitter":1, 
        "Join":6, 
        "Dearray":6, 
        "Rearray":8, 
        "Zip":node.datatype.length if getcn(node) == 'Zip' else 0,
        "Zippee":1,
        "Transpose":node.datatype.length*10 if getcn(node) == 'Transpose' else 0, 
        "Mather":(15 if node.kind == 'add' else 40)if getcn(node) == 'Mather' else 0, 
        "UserFun":i
    }
    ext = execution_times[getcn(node)] if getcn(node) in execution_times.keys() else 20
    
    phase_count = str(ext)+(',' + str(ext)) * (check_and_get_phasecount(node) - 1)
    properties = """<actorProperties actor='{0}'>
        <processor type='cluster_0' default='true'>
            <executionTime time='{1}'/>
        </processor>
    </actorProperties>
    """.format(node.name, phase_count)
    return properties

def get_csdf(g, i=50):
    nodes = g["nodes"]
    channels = g["channels"]
    
    # Build
    sdf = xml_hdr
    sdf += "\n".join([str(node) for node in nodes])
    sdf += "\n".join([str(channel) for channel in channels])
    sdf += xml_mid
    sdf += "\n".join([get_actor_properties(node, i) for node in nodes])
    sdf += xml_end
    
    return sdf


def add_cwd(fn):
    return os.getcwd() + "/" + fn

def write_csdf(g, filename, i=50):
    csdf = get_csdf(g, i)
    with open(filename, 'w') as f:
        f.write(csdf)
        

def read_program(filename):
    with open(filename) as f:
        return f.read()
    
    


# In[6]:



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
            
            
def get_sdf_from_hl(p):
    return get_sdf(smush_rede(rec_explode(unpack_program(p)["graph"])))

from Classes import *
