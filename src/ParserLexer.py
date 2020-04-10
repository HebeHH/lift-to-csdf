#!/usr/bin/env python
# coding: utf-8

# In[28]:


from Data import *


# In[36]:


# Parser


# Input: Program code as string
def unpack_program(program, sv_file):
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
            i1 =  Param(getname(), "fst"+new_node.name)
            i2 =  Param(getname(), "snd"+new_node.name)
            return {
                "nodes" : [new_node, i1, i2],
                "channels" : [get_channel(i1.new_outport(), new_node.new_inport()),
                             get_channel(i2.new_outport(), new_node.new_inport())],
                "inputs" : [i1.input, i2.input],
                "output" : new_node.output
            }
            
        new_node = UserFun(getname(), bby)
        bbyc = []
        bbyn = [new_node]
        bbyin = []
        for i in range(n_inputs):
            tmp =  Param(getname(), str(i))
            bbyc.append(get_channel(tmp.new_outport(), new_node.new_inport()))
            bbyin.append(tmp.input)
            bbyn.append(tmp)
        return {
            "nodes" : bbyn,
            "channels" : bbyc,
            "inputs" : bbyin,
            "output" : new_node.output
        }
        
    
    # Output: {nodes : [Node], channels : [{Port, Port}], inputs : [Port], output : Port}
    def unpack_func(func):
        nodes = []
        channels = []
        myinputs = []
        local_names = getnames(func) if "=>" in func else []
        subfuncs = {}

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
            if op in subfuncs.keys():
                args = getNextParams(comp)
                
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
#         print(func)
#         print(res)
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
    
#     TODO: SUBFUNCS
#     for sf in getsubfuncs(program):
#         subfuncs[sf[0]] = {'raw' : sf[1],
#                            'func' : commasplit(sf[1])[-1]
#                           }
    parsed = unpack_func(main_func)
    res = {
        "sizevars" : sizevars,
        "inputs" : starting_inputs,
        "code" : getfunc(program),
        "graph" : parsed
    }       
    if sv_file:
        if type(sv_file) is str:
            with open(sv_file) as f:
                sv_file = [int(x) for x in f.readlines()]
        cascader(res, lambda i: sv_file[i])
    else:
        cascader(res)
#     add_dt_to_channels(res)
    return res


# In[37]:


def parse_file(fn, sv_file = None):
    program = read_program(fn)
    p = unpack_program(program, sv_file)
    return p


# In[38]:


def test():
    return parse_file(add_cwd("highLevel/mmNN"))


# In[41]:


# p = test()
# parse_file(add_cwd("highLevel/mydot"))


# In[6]:


# printgraph_simple(p['graph'])


# In[8]:



type(1)


# In[ ]:




