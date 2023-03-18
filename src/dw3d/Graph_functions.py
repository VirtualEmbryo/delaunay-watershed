from time import time 
import numpy as np 
import networkx


def give_faces_table(Tetrahedrons):
    Faces_table =[]
    for i,tet in enumerate(Tetrahedrons): 
        a,b,c,d = tet
        Faces_table.append([a,b,c,i])
        Faces_table.append([a,b,d,i])
        Faces_table.append([a,c,d,i])
        Faces_table.append([b,c,d,i])
    return(Faces_table)

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)      

def lambdaSort_tri(Table,key_multiplier):
    Table.sort(key=lambda x:(x[0]+1)*(key_multiplier**3) + (x[1]+1)*(key_multiplier**2) + (x[2]+1)*(key_multiplier**1) + (x[3]+1))


def Faces_score_from_sampling(Faces, Verts,f): 

    alpha = np.linspace(0,1,5)[1:-1]
    beta = np.linspace(0,1,5)[1:-1]
    gamma = np.linspace(0,1,5)[1:-1]

    Vs = Verts.copy()
    scale = np.amax(Verts,axis=0)/2
    Vs-=scale
    Vs*=(1-1e-4)
    Vs+=scale*(1-1e-4)
    V = Vs[Faces]

    V1 = V[:,0]
    V2 = V[:,1]
    V3 = V[:,2]
    count = 0
    count_bad = 0
    Score_Faces = np.zeros(len(Faces))
    for a in (alpha) : 
        for b in beta : 
            for c in gamma :
                try : 
                    s = a+b+c 
                    l1 = a/s
                    l2 = b/s
                    l3 = c/s
                    
                    Score_Faces+=np.array(f(V1*l1+V2*l2+V3*l3))
                    count+=1
                except : 
                    count_bad+=1
    Score_Faces/=count
    return(Score_Faces)

class Delaunay_Graph(): 

    def __init__(self, tri,edt,labels,print_info = False):
        t1 = time()
        self.Nodes = tri.simplices
        self.Vertices = tri.points
        self.tri = tri
        self.n_simplices = len(tri.simplices)
        self.edt = edt
        self.labels = labels

        edges_table = self.construct_edges_table()
        self.construct_edges(edges_table)
        self.compute_scores(edt)
        t2 = time()
        if print_info : print("Graph build in ",np.round(t2-t1,3))

    def construct_edges_table(self): 
        Tetra = np.sort(self.tri.simplices,axis=1)
        self.Tetra = Tetra.copy()
        Tetra+=1 #We shift to get the right keys
        faces_table = np.array(give_faces_table(Tetra))
        key_multiplier = find_key_multiplier(max(len(self.tri.points),len(self.tri.simplices)))
        Keys = faces_table[:,0]*(key_multiplier**3)+faces_table[:,1]*(key_multiplier**2)+faces_table[:,2]*(key_multiplier**1) +faces_table[:,3]*(key_multiplier**0)
        edges_table=(faces_table[np.argsort(Keys)])#.tolist()

        return(edges_table)

    def construct_edges(self,edges_table): 
        index = 0 
        n = len(edges_table)

        self.Faces = []
        self.Nodes_Linked_by_Faces=[]
        self.Nodes_on_the_border=np.zeros(len(self.Nodes))
        self.Faces_of_Nodes = {}
        self.Lone_Faces=[]
        self.Nodes_linked_by_lone_faces=[]
        while index < n-1 : 
   
            if edges_table[index][0]==edges_table[index+1][0] and edges_table[index][1]==edges_table[index+1][1] and edges_table[index][2]==edges_table[index+1][2]: 
                a,b = edges_table[index][3],edges_table[index+1][3]
                self.Faces.append(edges_table[index][:-1]-1)  #We correct the previous shift
                self.Nodes_Linked_by_Faces.append([a,b])
                self.Faces_of_Nodes[a] = self.Faces_of_Nodes.get(a,[])+[len(self.Faces)-1]
                self.Faces_of_Nodes[b] = self.Faces_of_Nodes.get(b,[])+[len(self.Faces)-1]
                index+=2
            else : 
                self.Nodes_on_the_border[edges_table[index][3]]=1
                self.Lone_Faces.append(edges_table[index][:-1]-1)
                self.Nodes_linked_by_lone_faces.append(edges_table[index][3])
                index+=1

        

        self.Faces = np.array(self.Faces)
        self.Nodes_Linked_by_Faces = np.array(self.Nodes_Linked_by_Faces) 

        self.Lone_Faces = np.array(self.Lone_Faces)
        self.Nodes_linked_by_lone_faces = np.array(self.Nodes_linked_by_lone_faces)
    
    def construct_nodes_edges_list(self): 
        Nodes =np.zeros((len(self.Tetra),4),dtype=int)
        Indexes = np.zeros(len(self.Tetra),dtype=int)

        for i,pair in enumerate(self.Nodes_Linked_by_Faces) : 
            a,b= pair
            Nodes[a,Indexes[a]]=i+1
            Nodes[b,Indexes[b]]=i+1
            Indexes[a]+=1
            Indexes[b]+=1

        return(Nodes)
    
    def compute_scores(self,edt): 
        #Remember : each edge is a face ! 
        Scores = Faces_score_from_sampling(self.Faces, self.Vertices,edt)
        self.Scores = Scores

    def compute_volumes(self): 
        Pos = self.Vertices[self.Tetra]
        Vects = Pos[:,[0,0,0]]-Pos[:,[1,2,3]]
        Volumes = np.abs(np.linalg.det(Vects))/6
        return(Volumes)

    def compute_areas(self):
        #Triangles[i] = 3*2 array of 3 points of the plane
        #Triangles = self.Vertices[self.Faces]
        Pos = self.Vertices[self.Faces]
        Sides = Pos-Pos[:,[2,0,1]]
        Lengths_sides = np.linalg.norm(Sides,axis = 2)
        Half_perimeters = np.sum(Lengths_sides,axis=1)/2
        
        Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
        Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
        return(Areas)

    def compute_nodes_centroids(self): 
        return(np.mean(self.Vertices[self.Nodes],axis=1))
    
    def compute_zero_nodes(self):
        Centroids = self.compute_nodes_centroids()
        bools = self.labels(Centroids)==0
        ints = np.arange(len(Centroids))[bools]
        return(ints)

    def networkx_graph_weights_and_borders(self): 

        self.Volumes = self.compute_volumes()  #Number of nodes (Tetrahedras)
        self.Areas = self.compute_areas()  #Number of edges (Faces)

        G = networkx.Graph()
        nt = len(self.Volumes)
        Dicts = [{'volume':x} for x in self.Volumes]
        G.add_nodes_from(zip(np.arange(nt),Dicts))
    
        Indices=np.arange(len(self.Faces))
        network_edges = np.array([(self.Nodes_Linked_by_Faces[idx][0],self.Nodes_Linked_by_Faces[idx][1],{'score': self.Scores[idx],'area': self.Areas[idx]}) for idx in Indices])
        
        G.add_edges_from(network_edges)

        return G



