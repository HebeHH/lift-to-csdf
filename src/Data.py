#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Classes
import copy


# In[2]:


def get_all_nodes(graph):
    sub_nodes = flatten([get_all_nodes(node.subfunc) for node in graph['nodes'] if getcn(node).startswith(("Reduce", "Map", "Iterate"))])
    return graph['nodes'] + sub_nodes
def get_all_channels(graph):
    relevant_nodes = [node for node in graph['nodes'] if getcn(node).startswith(("Reduce", "Map", "Iterate"))]
    sub_nodes = flatten([get_all_channels(node.subfunc) for node in relevant_nodes])
    for node in relevant_nodes:
        sub_nodes += [get_channel(node.output.name, x) for x in node.subfunc['inputs']]
    return graph['channels'] + sub_nodes


def magic_sv(i):
    myopts = [30]
    return random.choice(myopts)
    


# In[3]:


def get_all_nodes(graph):
    sub_nodes = flatten([get_all_nodes(node.subfunc) for node in graph['nodes'] if getcn(node).startswith(("Reduce", "Map", "Iterate"))])
    return graph['nodes'] + sub_nodes
def get_all_channels(graph):
    sub_channels = flatten([get_all_channels(node.subfunc) for node in graph['nodes'] if getcn(node).startswith(("Reduce", "Map", "Iterate"))])
    return graph['channels'] + sub_channels


# In[4]:


def cascader(p, sv_maker = magic_sv):
    g = p['graph']
    ns = get_all_nodes(g)
    cs = get_all_channels(g)
    svs = dict([(p['sizevars'][i], sv_maker(i)) for i in range(len(p['sizevars']))])
    final_output = get_output_node(g).name
    cascade(p, ns, cs, svs, final_output )
    
def cascade(program, ns, cs, svs, final_output, triggers = {}):
    g = program["graph"]
    tmp_g = {'nodes':ns,'channels':cs}
    nd = dict([(n.name, n) for n in ns])
    cbsp = dict([(c.src_port, c) for c in cs])
    
    
    visited = []
    
    def form_type(s):
        if isinstance(s, TYPECLASS):
            return s
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
                l = svs[paras[1]] if 'SizeVar' not in paras[1] else magic_sv(0)
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
            elif cn.startswith(("Reduce", "Map")):
                print("SHOULDNT HAPPEN")
                cascaded_inputs = [(n.label, inc_dt.subdata) for n in get_input_nodes(node.subfunc)]
                bby_program = {'sizevars': program['sizevars'], 
                        'inputs': cascaded_inputs, 
                        'graph': node.subfunc}
                cascade(bby_program, ns, cs, svs)
                bby_outnode = get_output_node(bby_program['graph'])
                return bby_outnode.datatype
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
#                 print("DEARRAY ", node_num)
                return inc_dt.subdata # A(e, N) -> e
            elif cn == "Rearray":
                node_num = re.findall(r'[0-9]+', node.name)[0]
                new_len = 1 if node.masternode.startswith("Reduce") else dearray_by[node_num]
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
            
    
                cascaded_inputs = [(n.label, inc_dt.subdata) for n in get_input_nodes(node.subfunc)]
                bby_program = {'sizevars': program['sizevars'], 
                        'inputs': cascaded_inputs, 
                        'graph': node.subfunc}
                cascade(bby_program, ns, cs, svs)
                bby_outnode = get_output_node(bby_program['graph'])
    
    
#     print(final_output)
    def push(node):
#         print("Start:", node.name,getcn(node))
        assert(node.has_dt())
        my_dt = node.get_dt()
        if node in triggers:
            masternode = triggers[node]
            length = masternode.rep if getcn(masternode) == 'Map' else 1
            masternode.add_dt(Array(length, my_dt))
            push(masternode)
            return
        if node.name in visited or node.name == final_output:
            return
        visited.append(node.name)
#         print("Pushing:", node.name,getcn(node))
        
        output_nodes = [get_connected_node(tmp_g, out.name) for out in getout(node)]
        output_ports = getout(node)
        output_channels = [cbsp[p.name] for p in output_ports]
        [c.add_dt(my_dt) for c in output_channels]
        output_nodes2 = [nd[c.dst_act] for c in output_channels]
        assert(output_nodes == output_nodes2)
#         printls(output_nodes)
        
        for out_node in output_nodes:
#             print(node.name,"is pushing out node", out_node.name)
            inc_nodes = [get_connected_node(tmp_g, inc.name) for inc in getin(out_node)]
            if not all([x.has_dt() for x in inc_nodes] ):
#                 print("not all inputs")
                continue
            if getcn(out_node).startswith(("Map", "Reduce")):
#                 print("hello map", out_node.name)
                out_node.rep = my_dt.length
                cascaded_inputs = [(n.label, my_dt.subdata) for n in get_input_nodes(out_node.subfunc)]
                bby_program = {'sizevars': program['sizevars'], 
                        'inputs': cascaded_inputs, 
                        'graph': out_node.subfunc}
                bby_outnode = get_output_node(bby_program['graph'])
                triggers[bby_outnode] = out_node
#                 print(out_node.name, " trigger ",bby_outnode.name)
                cascade(bby_program, ns, cs, svs, final_output, triggers)
                continue
            if getcn(out_node) != 'Zip':
#                 print("isn't zip")
                new_dt = def_odt(out_node, my_dt)
                out_node.add_dt(new_dt)
                push(out_node)
            else:
#                 print("is zip")
                my_dts = [get_connected_node(tmp_g, inc.name).datatype for inc in getin(out_node)]
                new_dt = def_odt(out_node, my_dts)
                out_node.add_dt(new_dt)
                push(out_node)
    
    
    # set types of input paramaters:
    starters = []
#     print("Hello")
    for k, dt in in_types:
        param_nodes = [n for n in ns if getcn(n) == 'Param' and n.label == k]
        assert(len(param_nodes) == 1)
        param_node = param_nodes[0]
#         print('param node is ',k, param_node.name)
        param_node.add_dt(dt)
        starters.append(param_node)
#     print("goodbye")
#     print("pushing", [s.name for s in starters])
    for starter in starters:
        push(starter)
    return program

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


def add_dt_to_channels_g(g):
    cs = g["channels"]
    nd = dict([(n.name, n) for n in g["nodes"]])
    for c in cs:
        in_node = nd[c.src_act]
        in_dt = in_node.get_dt()
        c.add_dt(in_dt)
    return g

def add_datatypes(p, sizevar_file):
    sizevar_func = get_sizes_from_file(sizevar_file)
    cascade(p, sizevar_func)
    add_dt_to_channels(p)


# In[ ]:





# In[ ]:





# In[ ]:




