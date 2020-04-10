#!/usr/bin/env python
# coding: utf-8

# In[1]:


liftdir = "/Users/hebe/Dropbox/Yale-NUS/Yr4/capstone/forrealsies/lift"

import os, re
import pandas as pd
# from typing import NamedTuple

def flatten(lol):
    return [val for sublist in lol for val in sublist]

# class Node(NamedTuple):
#     name: str
#     kind: str
        
# class Edge(NamedTuple):
#     source: str
#     target: str
#     code: str
        
class Edge():
    def __init__(self, source :str, target :str, code :str):
        self.source = source
        self.target = target
        self.code = code
        pass
class EdgeList():
    def __init__(self, incoming: list, outgoing: list):
        self.incoming = incoming
        self.outgoing = outgoing
        pass
class Graph():
    def __init__(self, name: str, nodes : dict, edges : dict, info : dict ):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.info = info
        pass
# class EdgeList(NamedTuple):
#     incoming: list
#     outgoing: list

# class Graph(NamedTuple):
#     nodes: dict
#     edges: dict
        
def not_connected(el):
    return len(el.incoming) == 0 and len(el.outgoing) == 0

def edge_match(e1, e2):
    return e1.target == e2.target and e1.source == e2.source and e1.code == e2.code


# In[2]:


"poli()0i".replace("(","0")


# In[ ]:





# In[ ]:





# In[3]:


def read_and_format(file):

    # Read nodes and edges from program
    with open(file) as f:
        tmp = f.read()
    tmp = tmp.split('\n\n\n')
    nodes_raw = tmp[0].split('\n')[1:]
    edges_raw = tmp[1].split('\n')[1:-1]

    # Dictionary of extracted node name -> type mapping
    nodes = {}
    info = {}
    for node in nodes_raw:
        tmp = node.split(':')
        nodes[tmp[0].strip()] = re.sub(r'Lambda[0-9]+', 'Lambda', tmp[1]).strip()
        info[tmp[0].strip()] = ':'.join(tmp[2:])

    # Extract edge information
    edge_split = [(x.split(' -> ')[0].strip(), 
               x.split(' -> ')[1].split(':')[0].strip(), 
               re.sub(r' \(\)','',x.split(':')[1]).strip()) 
              for x in edges_raw]

    # Formatting edge information
    edges = {}
    for node in nodes.keys():
        inc = [Edge(e[0], e[1], e[2]) for e in edge_split if e[1] == node] 
        out = [Edge(e[0], e[1], e[2]) for e in edge_split if e[0] == node]
        edges[node] = EdgeList(inc, out)

    # Find and remove lone nodes
    # lone_nodes = [nodes.pop(n) for n in nodes.keys() if not_connected(edges[n])]
    lone_nodes = [n for n in nodes.keys() if not_connected(edges[n])]
    for lone_node in lone_nodes:
        nodes.pop(lone_node)
        info.pop(lone_node)

    # Deal with funcalls with duplicated hashcode
    # Current def of 'duplicated hashcode' is 'has multiple other nodes pointing to is as function'
    dupl_funcall = [n for n, v in nodes.items() 
                     if v == 'FunCall' and len(edges[n].incoming) > 1]
    
    # TODO: must remove ALL edges on both sides and also add mirrored edges
    def handle_weird_funcall(n):
        # find n n
        parents = [(e.source, e.code) for e in edges[n].incoming]
        count = len(parents)
        # separate and assert n copies of every kind of outgoing edge
        child_codes = set([e.code for e in edges[n].outgoing])
        child_edges = {}
        for c in child_codes:
            child_edges[c] = [e for e in edges[n].outgoing if e.code == c]
            assert(count == len(child_edges[c]))
        # create n duplicate nodes
        # assign each a parent and a set of outgoing edges
        for i in range(count):
            new_name = 'n' + str(len(nodes.keys()) + 2)
            nodes[new_name] = 'FunCall'
            inc = [Edge(parents[i][0], new_name, parents[i][1])]
            out = [Edge(new_name, child_edges[c][i], c) for c in child_codes]
            edges[new_name] = EdgeList(inc, out)
        # remove original node and all edges
        nodes.pop(n)
        edges.pop(n)
        info.pop(n)

    for df in dupl_funcall:
        handle_weird_funcall(df)
    
    name = file.split('/')[-1]
    return Graph(name, nodes, edges, info)


# In[4]:


funcall_f_targets = []

def assumptions_checker(graph):  
    nodes = graph.nodes
    edges = graph.edges
    
    # FunCall Assertions:
    funcalls = [n for n, v in nodes.items() if v == 'FunCall']
    for funcall in funcalls:
        # Each FunCall has excactly 1 incoming edge, 
        #      from either another FunCall or a Lambda, 
        #      as 'body' or 'arg_n'
        inc = edges[funcall].incoming
        assert len(inc) == 1, 'FunCall  has incorrect number of incoming edges' 
        assert inc[0].code != 'f', 'FunCall incedge is not "f"' 
        assert nodes[inc[0].source] in ['FunCall', 'Lambda'], 'FunCall incoming source is not a FunCall or Lambda'
        
        # Each FunCall has exactly one (outgoing edge labelled f)
        # All other edges are 'arg_n' from 0 to num edges - 2
        out = edges[funcall].outgoing
        assert len([e for e in out if e.code == 'f']) == 1 
        if len(out) > 1:
            codes = [int(x) for x in re.findall(r'arg_([0-9]+)', ' '.join([e.code for e in out]))]
            assert len(codes) == len(set(codes)), 'FunCall has duplicate arg codes' 
            assert min(codes) == 0, 'Funcall args do not start at 0' 
            assert max(codes) == len(out) - 2, 'FunCall args not incrementing correctly' 
#         print([nodes[e.target] for e in out if e.code == 'f'])

    
    # Lambda Assumptions
    lambdas = [n for n, v in nodes.items() if v == 'Lambda']
    for lam in lambdas:
        # Each Lambda has 0 or 1 incoming edges, from Map, Reduce or ReduceSeq, as 'f'
        inc = edges[lam].incoming
        assert len(inc) <= 1, 'Lambda has too many incoming edges' 
        if len(inc) > 0:
            assert nodes[inc[0].source] in ['Map', 'Reduce'],  'Lambda has incoming edge from odd source' 
            assert inc[0].code == 'f',  'Lambda has non-f incoming source' 
        
        # Each lambda has exactly 1 (outgoing edge labelled body) which points to a FunCall
        # All other outgoing edges are labelled 'param_n', n increasing from 0
        out = edges[lam].outgoing
        assert len([e for e in out if e.code == 'body']) == 1,  'Lambda does not have exactly one body' 
        assert len([e for e in out if e.code == 'body' and nodes[e.target] == 'FunCall']) == 1, 'Lambdas body is not a FunCall' 
        if len(out) > 1:
            codes = [int(x) for x in re.findall(r'param_([0-9]+)', ' '.join([e.code for e in out]))]
            assert len(codes) == len(set(codes)), 'Lambda has duplicate Param codes'
            assert min(codes) == 0, 'Lambda Params do not start at 0' 
            assert max(codes) == len(out) - 2, 'Lambda Params do not increment properly' 
    
    # Non-duplicate arg/param num assumptions:
    for n in nodes.keys():
        out = edges[n].outgoing
        if len(out) > 1:
            codes = [int(x) for x in re.findall(r'_([0-9]+)', ' '.join([e.code for e in out]))]
            assert len(codes) == len(set(codes)), 'Node has duplicate parameter values'
            assert min(codes) == 0, 'Node parameter nums do not start at 0' 
            assert max(codes) < len(out), 'Node parameters do not increment properly' 
    
    # All nodes are connected
    lone_nodes = [n for n in nodes.keys() if not_connected(edges[n])]
    assert len(lone_nodes) == 0, "Disconnected node in graph"
    
    # Params don't point anywhere
    for param in [n for n, v in nodes.items() if v == 'Param']:
        assert len(edges[param].outgoing) == 0, "Params don't point anywhere"
    
    # All edges reference real nodes
    for k, v in edges.items():
        for e in v.incoming:
            assert e.source in nodes and e.target in nodes, "rootless edge"
            assert any([edge_match(x,e) for x in edges[e.source].outgoing]), "non-mirrored edge"
        for e in v.outgoing:
            assert e.source in nodes and e.target in nodes, "rootless edge"
            assert any([edge_match(x,e) for x in edges[e.target].incoming]), "non-mirrored edge"
#             assert e in edges[e.target].incoming, "non-mirrored edge"
            
    
    return True

def assert_assumptions(graph):
    try:
        return assumptions_checker(graph)
    except AssertionError as e:
        return str(e)
    return False


# In[5]:


def drop_wrappers(graph):
    nodes = graph.nodes
    edges = graph.edges
    
    # FunCalls:
    for funcall in [n for n, v in nodes.items() if v == 'FunCall']:
        f_node = [e.target for e in edges[funcall].outgoing if e.code == 'f'][0]
        new_edges = [Edge(f_node, e.target, e.code) for e in edges[funcall].outgoing if e.code != 'f']
        parent = edges[funcall].incoming[0].source
        parent_edge = Edge(parent, f_node, edges[funcall].incoming[0].code)
        # Attach all args to f node
        edges[f_node].outgoing = edges[f_node].outgoing + new_edges
        # Attach f node to args
        for e in new_edges:
            edges[e.target].incoming = edges[e.target].incoming + [e]
        # Attach f node to parent
        edges[parent].outgoing = edges[parent].outgoing +[parent_edge]
        # Attach parent to f node:
        edges[f_node].incoming = edges[f_node].incoming +[parent_edge]
        # delete self and all edges
        for edge in edges[funcall].outgoing:
            edges[edge.target].incoming = list(filter(lambda e: e.source != funcall, edges[edge.target].incoming))
        edges[parent].outgoing = list(filter(lambda e: e.target != funcall, edges[parent].outgoing))
        nodes.pop(funcall)
        edges.pop(funcall)
        graph.info.pop(funcall)


    # Lambda:
    for lam in [n for n, v in nodes.items() if v == 'Lambda']:
        # attach body node to parent and vice versa
        if len(edges[lam].incoming) > 0:
            parent = edges[lam].incoming[0].source
            body_node = [e.target for e in edges[lam].outgoing if e.code == 'body'][0]
            parent_edge = Edge(parent, body_node, edges[lam].incoming[0].code)
            edges[parent].outgoing = edges[parent].outgoing + [parent_edge]
            edges[body_node].incoming = edges[body_node].incoming + [parent_edge]
        # delete self and edges
        for edge in edges[lam].outgoing:
            edges[edge.target].incoming = list(filter(lambda e: e.source != lam, edges[edge.target].incoming))
        edges[parent].outgoing = list(filter(lambda e: e.target != lam, edges[parent].outgoing))
        edges.pop(lam)
        nodes.pop(lam)
        graph.info.pop(lam)
        
    return True


# In[6]:


def make_dotfile(graph, filename):
    nodes = graph.nodes
    edges = graph.edges
    
    file = open(filename, 'w')
    file.write('digraph{ \n ratio="compress" \n size=8 \n margin="0.0,0.0"\n')
    
    for name, func in nodes.items():
        file.write(re.sub( r'[\(\)]' ,'',name) + " [style=rounded,shape=box,label=<<b>"+func+"</b>>]\n")
    for _, edgelist in edges.items():
        for edge in edgelist.incoming:
            file.write(re.sub( r'[\(\)]' ,'',edge.source) + ' -> ' + re.sub( r'[\(\)]' ,'',edge.target) + ' [label="' + re.sub( r'[\(\)]' ,'',edge.code) + '"];\n')
    file.write('}\n')


# In[7]:


def post_adaptions_assertions(graph):
    nodes = graph.nodes
    edges = graph.edges
    
    # prior expectations still met
    assumptions_checker(graph)
    
    for node in nodes:
        assert len(edges[node].incoming) < 2, 'All nodes only feed at most one other node'
    
    
    return True


# In[8]:


def collapse_userfuncs(graph):
    nodes = graph.nodes
    edges = graph.edges
    
    userfuncs = [n for n, v in nodes.items() if v == 'UserFun']
    
    def desc(node):
        return flatten([[e.target] + desc(e.target) for e in edges[node].outgoing])
#         return [node] + flatten([desc(e.target) for e in edges[node].outgoing])
    
    for uf in userfuncs:
        bundle = set(desc(uf))
#         print(bundle)
        if all([(nodes[n] in ['Param', 'UserFun'] or nodes[n].startswith('Value') or nodes[n].startswith('Get')) for n in bundle]):
            for node in bundle:
                if nodes[node] == 'Param':
                    farinc = [e for e in edges[node].incoming if e.source not in bundle]
                    if len(farinc) > 0:
                        edges[node].incoming = farinc
                    else:
                        nodes.pop(node)
                        edges.pop(node)
                        graph.info.pop(node)
                else:
                    nodes.pop(node)
                    edges.pop(node)
                    graph.info.pop(node)
            edges[uf].outgoing = []
        else:
            print(bundle)
            print([nodes[n] for n in bundle])
    return True          
                        
        
    


# In[9]:


def duplicate_params(graph):
    nodes = graph.nodes
    edges = graph.edges
    info = graph.info
    
    params = [n for n, v in nodes.items() if v == 'Param' and len(edges[n].incoming) > 1]
    
    #TODO: more foolproof system for coming up with new names of nodes
    for param in params: 
        for parent in [e.source for e in edges[param].incoming]:
            new_name = 'n' + str(len(nodes.keys()) + 40)
            nodes[new_name] = 'Param'
            info[new_name] = info[param]
            new_edge = Edge(parent, new_name, [e.code for e in edges[parent].outgoing if e.target == param][0])
            edges[new_name] = EdgeList([new_edge], [])
            edges[parent].outgoing = [new_edge] + [e for e in edges[parent].outgoing if e.target != param]
    return True


# In[10]:


# Must come last from current selection
# def reversio(graph):
#     edges = graph.edges
#     for node in edges.keys():
#         edges[node] = EdgeList(edges[node].outgoing, edges[node].incoming)
#     print('hi')
#     return True


# In[11]:


printeddir = liftdir + "/printed"
allfs =  os.listdir(printeddir)
allfs = [os.path.join(printeddir,x) for x in allfs if 
            os.path.isfile(os.path.join(printeddir,x)) and x.endswith('v2.txt')]

graphs = []

for file in allfs:
    graph = read_and_format(file)
#     print(file, assert_assumptions(graph))
    graphs.append(graph)

working_graphs = [graph for graph in graphs if assert_assumptions(graph) == True]
print("Started with: ", len(graphs), "graphs. \nPassed checks:", len(working_graphs))
# print("\nFailed: \n", '\n'.join([f + ":\n    Cause: " + assert_assumptions(g) for (g, f) in zip(graphs, allfs) if assert_assumptions(g) != True]))

streamlined_graphs = [graph for graph in working_graphs if drop_wrappers(graph)]

working_streamlined_graphs = [graph for graph in graphs if assert_assumptions(graph) == True]
print("\n\nDropping wrappers starting with: ", len(streamlined_graphs), "graphs. \nPassed checks:", len(working_streamlined_graphs))

for graph, fn in zip(graphs, allfs):
    fn = fn.split('.')[0] + '.dot'
    make_dotfile(graph, fn)
#     print("\ndot -Tpdf "+fn+" -o "+fn.split('.')[0]+".pdf")
    

# TODO: Fix duplicate params/collapse userfuncs
curr = [g for g in working_streamlined_graphs if collapse_userfuncs(g)]
# curr = [g for g in curr if duplicate_params(g)]

# print("\nCollapsed Userfuncs and expanded params")
print("Passed checks:", len([1 for g in graphs if assert_assumptions(graph) == True]))

for graph, fn in zip(graphs, allfs):
    fn = fn.split('.')[0] + 'woUFC.dot'
    make_dotfile(graph, fn)
#     print("\ndot -Tpdf "+fn+" -o "+fn.split('.')[0]+".pdf")


# In[12]:


g = graphs[0]
ni = g.info['n1(1)']
c = [s.replace('}','').strip() for s in ni.split('{')]
[s for s in c if s != 'ParamUnknown']


# In[13]:


# set(flatten(flatten([[clean_type(info) for k, info in g.info.items() if g.nodes[k] == 'Param'] for g in graphs])))

dot_graph = graphs[5]


# In[25]:


dot_graph.nodes


# In[24]:


set(flatten([g.nodes.values() for g in working_streamlined_graphs]))


# In[16]:


class Data:
    def __init__(self):
        pass

class Undef(Data):
    pass

class Unknown(Data):
    pass

class Tuple(Data):
    def __init__(self, length, subdatals = []):
        self.length = length
        self.subdatals = subdatals
        
    def add_subdata(subdata):
        self.subdatals += subdata

class Scalar(Data):
    def __init__(self, kind = 'float', size : int = 4):
        self.size = 4
        self.kind = kind

class Vector(Data):
    def __init__(self, length, subdata):
        self.length = length
        self.subdata = subdata
        
class Array(Data):
    def __init__(self, length, subdata):
        self.length = length
        self.subdata = subdata


# In[17]:


def clean_type(info):
    c = [s.replace('}','').strip() for s in info.split('{')]
    return [s for s in c if s != 'ParamUnknown']

def get_type(typelist):
    head = typelist[0]
    print(head)
    if head.startswith('Scalar'):
        assert len(typelist) == 1, 'Scalar is last'
        kind = re.findall(r'Scalar ([a-zA-Z]+) ', head)[0]
        size = re.findall(r'size = ([0-9a-zA-Z]+)', head)[0]
        return Scalar(kind, size)
    if head == 'UndefType':
        return Undef()
    if head.startswith('Vector'):
        length = re.findall(r'len = ([0-9a-zA-Z]+)', head)[0]
        subtype = get_type(typelist[1:])
        return Vector(length, subtype)
    if head.startswith('Array'):
        length = re.findall(r'len = ([0-9a-zA-Z]+)', head)[0]
        subtype = get_type(typelist[1:])
        return Array(length, subtype)
    if head.startswith('Tuple'):
        tuple_types =[x.strip() for x in typelist[1].split('-')]
        ret_tuple = Tuple(len(tuple_types))
        for x in tuple_types:
            ret_tuple.add_subdata(get_type([x]))
        return ret_tuple


# In[18]:


get_type(clean_type(ni)).subdata.subdata


# In[ ]:


{'Get (0)',
 'Map',
 'Param',
 'Reduce', 
 'Transpose',
 'UserFun', 
 'Value', 
 'Zip'}

Param
Value

UserFun

Zip
Reduce
Map


# In[3]:


ls = [0,1,2,3]
ls.append(8)
ls


# In[ ]:





# In[ ]:


nt = 'Param'

if nt == 'Param':
    


# In[19]:



class Port:
    def __init__ (self, direction, name, rate, dearray = False):
        self.name = name
        self.direction = direction
        self.rate = rate
        self.dearray = dearray
    def __str__(self):
        return "<port type='"+self.direction+"' name='"+self.name+"' rate='"+','.join(map(str,self.rate))+"'/>"

class Param():
    def __init__(self, name : str, info):
        self.name = name
        self.data = get_type(clean_type(info))
        self.output = [Port('out', name + '_out', [1])]
    def __str__(self):
        return "<actor name='"+self.name+"' type='Param'>\n" + '\n'.join(map(str, self.output)) + "\n</actor>"

class Value():
    def __init__(self, name : str, value, er = 0):
        self.name = name
        self.value = value
        self.empty_repeats = er
        self.output = [Port('out', name + '_out', [1] + ([0] * er))]
    def __str__(self):
        return "<actor name='"+self.name+"' type='Param'>\n" + '\n'.join(map(str, self.output)) + "\n</actor>"

class Reduce:
    def __init__(self, name : str):
        self.name = name
#         self.subfunc = subfunc
        self.input = [Port('in', name + '_in', [1], True)]
        self.output = [Port('out', name + '_out', [1], True)]
    def addSingleIteration(sg):
        self.single_iteration = sg
    def __str__(self):
        return "<actor name='"+self.name+"' type='Reduce'>\n" + '\n'.join(map(str, self.output ++ self.input)) + "\n</actor>"


class Map:
    def __init__(self, name : str):
        self.name = name
#         self.subfunc = subfunc
        self.input = [Port('in', name + '_in', [1], True)]
        self.output = [Port('out', name + '_out', [1], True)]
    def addSubgraph(sg):
        self.subgraph = sg
        self.input = sg.inputs[0]
        self.output = sg.output
        
    def __str__(self):
        return "<actor name='"+self.name+"' type='Map'>\n" + '\n'.join(map(str, self.output ++ self.input)) + "\n</actor>"
    
class Zip:
    def __init__(self, name : str):
        self.name = name
        self.input = [Port('in', name + '_in1', [1], True),Port('in', name + '_in2', [1], True)]
        self.output = [Port('out', name + '_out', [1], True)]
    def __str__(self):
        return "<actor name='"+self.name+"' type='Zip'>\n" + '\n'.join(map(str, self.output ++ self.input)) + "\n</actor>"

class UserFun:
    def __init__(self, name : str):
        self.name = name
        self.subfunc = subfunc
        self.input = [Port('in', name + '_in', [1])]
        self.output = [Port('out', name + '_out', [1])]
    def __str__(self):
        return "<actor name='"+self.name+"' type='UserFun'>\n" + '\n'.join(map(str, self.output ++ self.input)) + "\n</actor>"


# In[27]:


dot_graph.nodes
# .replace('(', 'X').replace(')', 'X')


# In[20]:


<?xml version="1.0" encoding="UTF-8"?>
<sdf3 type="csdf" version="1.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://www.es.ele.tue.nl/sdf3/xsd/sdf3-csdf.xsd">
<applicationGraph name='autogen'>
    <csdf name='autogen' type='autogen'>
        <actor name='n1X1X' type='Param'>
            <port type='out' name='n1X1X_out' rate='1'/>
        </actor>
        <actor name='n5X1X1' type='Dearray'>
            <port type='in' name='n5X5X1_in' rate='1'/>
            <port type='out' name='n5X5X1_out' rate='10'/>
        </actor>
        <actor name='n7X1X' type='UserFun'>
            <port type='in' name='n7X1X_in' rate='1'/>
            <port type='out' name='n7X1X_out' rate='1'/>
        </actor>
        <actor name='N10X1X' type='UserFun'>
            <port type='in' name='N10X1X_in' rate='1,1,1,1,1,1,1,1,1,1'/>
            <port type='out' name='N10X1X_out' rate='0,0,0,0,0,0,0,0,0,1'/>
            <port type='in'  name='N10X1X_self_in' rate='1,1,1,1,1,1,1,1,1,1'/>
            <port type='out' name='N10X1X_self_out' rate='1,1,1,1,1,1,1,1,1,0'/>
        </actor>
        <actor name='n8X1X2' type='Rearray'>
            <port type='in' name='n8X1X2_in' rate='10'/>
            <port type='out' name='n8X1X2_out' rate='1'/>
        </actor>
        <actor name='OUTPUT' type='Output'>
            <port type='in' name='OUTPUT' rate='1'/>
        </actor>
        <channel name='channel_0' srcActor='n1X1X' srcPort='n1X1X_out' dstActor='n5X1X1' dstPort='n5X5X1_in' size='1' initialTokens='0'/>
        <channel name='channel_1' srcActor='n5X1X1' srcPort='n5X1X1_out' dstActor='n7X1X' dstPort='n7X1X_08t' size='1' initialTokens='0'/>
        <channel name='channel_2' srcActor='n7X1X' srcPort='n7X1X_out' dstActor='N10X1X' dstPort='N10X1X_in' size='1' initialTokens='0'/>
        <channel name='channel_3' srcActor='N10X1X' srcPort='N10X1X_self_out' dstActor='N10X1X' dstPort='N10X1X_self_in' size='1' initialTokens='1'/>
        <channel name='channel_4' srcActor='N10X1X' srcPort='N10X1X_out' dstActor='n8X1X2' dstPort='n8X1X2_in' size='1' initialTokens='0'/>
        <channel name='channel_5' srcActor='n8X1X2' srcPort='n8X1X2_out' dstActor='OUTPUT' dstPort='OUTPUT' size='1' initialTokens='0'/>
    </csdf>

    <csdfProperties>
        <actorProperties actor='n1X1X'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1'/>
            </processor>
        </actorProperties>
        <actorProperties actor='n5X1X1'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1'/>
            </processor>
        </actorProperties>
        <actorProperties actor='n7X1X'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1'/>
            </processor>
        </actorProperties>
        <actorProperties actor='N10X1X'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1,1,1,1,1,1,1,1,1,1'/>
            </processor>
        </actorProperties>
        <actorProperties actor='n8X1X2'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1'/>
            </processor>
        </actorProperties>
        <actorProperties actor='OUTPUT'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1'/>
            </processor>
        </actorProperties>
    </csdfProperties>

</applicationGraph>

</sdf3>


# In[ ]:





# In[ ]:


.replace('(', 'X').replace(')', 'X')


<?xml version="1.0" encoding="UTF-8"?>
<sdf3 type="csdf" version="1.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://www.es.ele.tue.nl/sdf3/xsd/sdf3-csdf.xsd">
<applicationGraph name='autogen'>
    <csdf name='autogen' type='autogen'>
        <actor name='A' type='a'>
            <port type='in' name='out_channel_3' rate='1,3'/>
            <port type='out' name='in_channel_1' rate='3,5'/>
            <port type='in'  name='out_RA' rate='1,1'/>
            <port type='out' name='in_RA' rate='1,1'/>
        </actor>
        <actor name='B' type='a'>
            <port type='in' name='out_channel_1' rate='1,1,4'/>
            <port type='out' name='in_channel_2' rate='6,2,1'/>
            <port type='in'  name='out_RB' rate='1,1,1'/>
            <port type='out' name='in_RB' rate='1,1,1'/>
        </actor>
        <actor name='C' type='a'>
            <port type='in' name='out_channel_2' rate='6'/>
            <port type='out' name='in_channel_3' rate='2'/>
            <port type='in'  name='out_RC' rate='1'/>
            <port type='out' name='in_RC' rate='1'/>
        </actor>
        <channel name='channel_A' srcActor='A' srcPort='in_RA' dstActor='A' dstPort='out_RA' size='1' initialTokens='1'/>
        <channel name='channel_B' srcActor='B' srcPort='in_RB' dstActor='B' dstPort='out_RB' size='1' initialTokens='1'/>
        <channel name='channel_C' srcActor='C' srcPort='in_RC' dstActor='C' dstPort='out_RC' size='1' initialTokens='1'/>


        <channel name='channel_1' srcActor='A' srcPort='in_channel_1' dstActor='B' dstPort='out_channel_1' size='1' initialTokens='0'/>
        <channel name='channel_2' srcActor='B' srcPort='in_channel_2' dstActor='C' dstPort='out_channel_2' size='1' initialTokens='0'/>
        <channel name='channel_3' srcActor='C' srcPort='in_channel_3' dstActor='A' dstPort='out_channel_3' size='1' initialTokens='4'/>
    </csdf>

    <csdfProperties>
        <actorProperties actor='A'>
            <processor type='cluster_0' default='true'>
                <executionTime time='3,1'/>
            </processor>
        </actorProperties>
        <actorProperties actor='B'>
            <processor type='cluster_0' default='true'>
                <executionTime time='2,1,2'/>
            </processor>
        </actorProperties>
        <actorProperties actor='C'>
            <processor type='cluster_0' default='true'>
                <executionTime time='1'/>
            </processor>
        </actorProperties>
    </csdfProperties>

</applicationGraph>

</sdf3>

