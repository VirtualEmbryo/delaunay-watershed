import numpy as np 
import networkx



def give_edges_table(Faces):
    #gives a table with a,b,ind_face

    Edges_table =[]
    for i,face in enumerate(Faces): 
        a,b,c = face
        Edges_table.append([a,b,i])
        Edges_table.append([a,c,i])
        Edges_table.append([b,c,i])
    return(Edges_table)

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)



class Delaunay_Graph(): 

    def __init__(self, tri,edt,labels,nx,ny,npoints=10):
        self.Nodes = tri.simplices
        self.Vertices = tri.points
        self.tri = tri
        self.n_simplices = len(tri.simplices)
        self.edt = edt
        self.labels = labels
        self.nx = nx
        self.ny = ny

        edges_table = self.construct_edges_table()
        self.construct_edges(edges_table)
        self.compute_scores(edt,npoints)


    def construct_edges_table(self): 
        Tris = np.sort(self.tri.simplices,axis=1)
        self.Tris = Tris.copy()
        Tris+=1 #We shift the indices to avoid 0 index. 
        edges_table = np.array(give_edges_table(Tris))
        key_multiplier = find_key_multiplier(max(len(self.tri.points),len(self.tri.simplices)))
        Keys = edges_table[:,0]*(key_multiplier**2)+edges_table[:,1]*(key_multiplier**1)+edges_table[:,2]
        edges_table=(edges_table[np.argsort(Keys)])

        return(edges_table)

    def construct_edges(self,edges_table): 
        index = 0 
        n = len(edges_table)

        self.Edges = []
        self.Nodes_linked_by_Edges=[]
        self.Lone_edges=[]
        self.Nodes_linked_by_lone_edges=[]
        while index < n-1 : 

            if edges_table[index][0]==edges_table[index+1][0] and edges_table[index][1]==edges_table[index+1][1] : 
                a,b = edges_table[index][2],edges_table[index+1][2]
                self.Edges.append(edges_table[index][:-1]-1) #We shift back the indices
                self.Nodes_linked_by_Edges.append([a,b])
                index+=2
            else : 
                self.Lone_edges.append(edges_table[index][:-1]-1)
                self.Nodes_linked_by_lone_edges.append(edges_table[index][2])
                index+=1

        self.Edges = np.array(self.Edges)
        self.Nodes_linked_by_Edges = np.array(self.Nodes_linked_by_Edges) 
        self.Lone_edges = np.array(self.Lone_edges)
        self.Nodes_linked_by_lone_edges = np.array(self.Nodes_linked_by_lone_edges)
    def construct_nodes_edges_list(self): 
        Nodes =np.zeros((len(self.Tris),3),dtype=int)
        Indexes = np.zeros(len(self.Tris),dtype=int)

        for i,pair in enumerate(self.Nodes_linked_by_Edges) : 
            a,b= pair
            Nodes[a,Indexes[a]]=i+1
            Nodes[b,Indexes[b]]=i+1
            Indexes[a]+=1
            Indexes[b]+=1

        return(Nodes)

    def compute_scores(self,edt,npoints): 
        """Score of every dual-graph edge: the mean of the EDT at ``npoints`` points on the
        Delaunay edge, endpoints included (pixel units).

        Vectorised over edges; the samples are accumulated in the same order and with the same
        weights as the original per-edge loop, so the scores are bitwise identical to it.
        ``edt`` must provide ``at(points)`` for an ``(n, 2)`` array of ``(row, col)`` points
        (``BilinearOnPixelGrid``); an interpolant without it falls back to one call per point.
        """
        sampling_parts = np.linspace(0,1,npoints)
        sampling_scores = np.ones(len(sampling_parts))/len(sampling_parts)
        Verts = self.Vertices.copy()[:,[0,1]]

        Scores=np.zeros(len(self.Edges))
        if len(self.Edges) == 0 or not hasattr(edt, "at"):
            for idx,edge in enumerate(self.Edges) : 
                for i,value in enumerate(sampling_parts) : 
                    point = Verts[edge[0]]*value + Verts[edge[1]]*(1-value)
                    Scores[idx] +=sampling_scores[i]*edt(point[0],point[1])[0]
            self.Scores = np.array(Scores)
            return

        first, second = Verts[self.Edges[:, 0]], Verts[self.Edges[:, 1]]
        for i,value in enumerate(sampling_parts) : 
            points = first*value + second*(1-value)
            Scores +=sampling_scores[i]*edt.at(points)
    
        self.Scores = np.array(Scores)


    def compute_areas(self): 
        Pos = self.Vertices[self.Tris]
        Vects = Pos[:,[0,0]]-Pos[:,[1,2]]
        Areas = np.abs(np.linalg.det(Vects))/2
        return(Areas)

    def compute_lengths(self):
        #Pos[i] = 2*2 array of 2 points of the plane
        Pos = self.Vertices[self.Edges]
        Sides = Pos[:,0]-Pos[:,1]
        Lengths_sides = np.linalg.norm(Sides,axis = 1)
        return(Lengths_sides)


    def compute_nodes_centroids(self): 
        return(np.mean(self.Vertices[self.Nodes],axis=1))
    
    
    def compute_zero_nodes(self):
        Centroids = self.compute_nodes_centroids()
        if hasattr(self.labels, "at"):
            bools = self.labels.at(Centroids)==0
        else:
            def func_labels(x): 
                return(self.labels(x[0],x[1]))
            bools = np.array(list(map(func_labels,Centroids)))[:,0]==0
        ints = np.arange(len(Centroids))[bools]
        return(ints)

    def networkx_graph_weights_and_borders(self, print_info=False): 

        Areas = self.compute_areas()  #Number of nodes (Triangles)
        Lengths = self.compute_lengths()  #Number of edges (Sides)

        G = networkx.Graph()
        nt = len(Areas)
        Dicts = [{'area':x} for x in Areas]
        G.add_nodes_from(zip(np.arange(nt),Dicts))
    
        Indices=np.arange(len(self.Edges))
        
        network_edges = np.array([(self.Nodes_linked_by_Edges[idx][0],self.Nodes_linked_by_Edges[idx][1],{'score': self.Scores[idx],'length':Lengths[idx]}) for idx in Indices])
        
        G.add_edges_from(network_edges)

        if print_info:  # printed unconditionally before 0.8.0
            print("Number of Nodes :",len(G.nodes)," Number of Edges :",len(G.edges))

        return G