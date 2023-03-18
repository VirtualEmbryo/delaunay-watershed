import networkx
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




def Assemble_Edges(Graph): 
    #See the definition of the different attributes in the method networkx_graph_weights_and_borders of the class Graph
    New_Graph = networkx.Graph()
    Edges=[]
    for node in Graph.nodes : 
        for neighbor in Graph.neighbors(node): 
            if node<neighbor : 
                
                weight=0
                area = 0
                
                num_edges= Graph.number_of_edges(node,neighbor)

                for key in range(num_edges): 
                    
                    area += Graph[node][neighbor][key]['area']
                    weight += Graph[node][neighbor][key]['score']*Graph[node][neighbor][key]['area']

                score = weight/area
                Edges.append((node,neighbor,{'score': score,'area':area}))
            
    New_Graph.add_nodes_from(Graph.nodes.data())
    New_Graph.add_edges_from(Edges)
    return(New_Graph)

#####
#SEEDED WATERSHED
#####


def seeded_watershed_aggregation(Graph,seeds,indices_labels): 
    #Seeds are expressed as labels of the nodes
    Labels = np.zeros(len(Graph.nodes),dtype=int)-1

    for i,seed in enumerate(seeds) : 
        Labels[seed]=indices_labels[i]
        
    Groups={}
    Number_Group=np.zeros(len(Graph.nodes),dtype=int)-1
    num_group = 0
    
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

def seeded_watershed_map(Graph,seeds,indices_labels,zero_nodes = []): 
    Labels = seeded_watershed_aggregation(Graph,seeds,indices_labels)
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








