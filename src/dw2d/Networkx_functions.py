import numpy as np

#####
#GENERAL
#PURPOSE
#FUNCTIONS
#####

def Reverse_map(Map): 
    #The reverse map is a dictionnary where reverse_map[ind]=connected_component to which belongs the ind
    reverse_map = {}
    for key in Map:
        for a in Map[key]: 
            reverse_map[a]=key
    return(reverse_map)


def seeded_watershed_aggregation(Graph,seeds,indices_labels,deterministic=True): 
    """Seeded watershed on the dual graph: edges visited by decreasing score, merging unlabelled groups.

    ``deterministic`` (default True): the edges are listed in canonical order, each as
    (smaller node, larger node) and the list sorted, and visited by decreasing score with a stable
    sort, so edges of equal score are met in that canonical order on every machine and every NumPy
    version. False: dw2d 0.1.5's order, the iteration order
    of the networkx graph and NumPy's default (unstable) sort, whose ties depend on the NumPy
    version and the CPU (27 of 237 meshes of the equilibrium-foam benchmark).
    """
    #Seeds are expressed as labels of the nodes
    Labels = np.zeros(len(Graph.nodes),dtype=int)-1
    for i,seed in enumerate(seeds) : 
        Labels[seed]=indices_labels[i]

    Groups={}
    Number_Group=np.zeros(len(Graph.nodes),dtype=int)-1
    num_group = 0
    
    if deterministic:
        Edges = sorted((min(a, b), max(a, b)) for a, b in Graph.edges)
        scores = np.array([Graph.edges[a, b]['score'] for a, b in Edges], dtype=float)
        args = np.argsort(-scores, kind="stable")
    else:
        args = np.argsort(-np.array(list(Graph.edges.data('score')))[:,2])
        Edges = list(Graph.edges)
    for arg in args: 
        a,b = Edges[arg]
        if Labels[a]!=-1 and Labels[b]!=-1 : 
            continue
        elif Labels[a]!=-1 and Labels[b]==-1 : 
            group = Groups.get(Number_Group[b],[b])
            Labels[group]=Labels[a]
        elif Labels[b]!=-1 and Labels[a]==-1 : 
            group = Groups.get(Number_Group[a],[a])
            Labels[group]=Labels[b]
        else : 
            if Number_Group[a]!=-1 : 
                if Number_Group[a]==Number_Group[b] : 
                    continue
                elif Number_Group[b]!=-1 : 
                    old_b_group = Groups.pop(Number_Group[b])
                    Groups[Number_Group[a]]+=old_b_group
                    Number_Group[old_b_group]=Number_Group[a]
                else : 
                    Groups[Number_Group[a]].append(b)
                    Number_Group[b]=Number_Group[a]
            else : 
                if Number_Group[b]!=-1 : 
                    Groups[Number_Group[b]].append(a)
                    Number_Group[a]=Number_Group[b]
                else : 
                    Number_Group[a]=num_group
                    Number_Group[b]=num_group
                    Groups[num_group]=[a,b]
                    num_group+=1
    return(Labels)

def seeded_watershed_map(Graph,seeds,indices_labels,zero_nodes = [],deterministic=True): 
    Labels = seeded_watershed_aggregation(Graph,seeds,indices_labels,deterministic=deterministic)
    return(apply_zero_and_seed_overrides(Labels,seeds,indices_labels,zero_nodes))


def apply_zero_and_seed_overrides(Labels,seeds,indices_labels,zero_nodes = []):
    """The watershed's last step: zero-node triangles take the medium (0), seed triangles their seed's label.

    ``Labels`` (one label per triangle) is modified in place; returns the map {label: [triangle ids]}.
    """
    Labels[zero_nodes]=0
    for i,seed in enumerate(seeds) : 
        Labels[seed]=indices_labels[i]

    Map_end = build_Map_from_labels(Labels)
    return(Map_end)


def build_Map_from_labels(PHI): 
    Map_end ={}
    for idx, label in enumerate(PHI) : 
        Map_end[label] = Map_end.get(label,[])
        Map_end[label].append(idx)
    return(Map_end)





