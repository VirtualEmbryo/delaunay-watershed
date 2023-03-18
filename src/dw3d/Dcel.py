#Sacha Ichbiah, Sept 2021
"""Module dedicated to the computation of geometrical quantities on a 3D Mesh."""

from dataclasses import dataclass, field
import math
import pickle
import numpy as np 
from dw3d.Curvature import compute_curvature_interfaces
from dw3d.Geometry import compute_areas_faces,compute_areas_cells,compute_angles_tri,compute_volume_cells, compute_volume_derivative_dict, compute_areas_interfaces,compute_area_derivative_dict, compute_length_trijunctions
import networkx



def separate_faces_dict(Faces,n_towers=10): 
    n_towers = np.amax(Faces[:,[3,4]])+1
   
    Occupancy=np.zeros(n_towers)
    Dict={}
    for face in Faces : 
        _,_,_,num1,num2 = face
        if num1!=-1:
            if Occupancy[num1]==0:
                Dict[num1]=[face[[0,1,2]]]
                Occupancy[num1]+=1
            else : 
                Dict[num1].append(face[[0,1,2]])
            
        if num2!=-1:
            if Occupancy[num2]==0:
                Dict[num2]=[face[[0,1,2]]]
                Occupancy[num2]+=1
            else : 
                Dict[num2].append(face[[0,1,2]])
            
            
    Faces_separated={}
    for i in sorted(Dict.keys()) : 
        Faces_separated[i] = (np.array(Dict[i]))
        
    return(Faces_separated)


def renormalize_verts(Verts,Faces): 
    #When the Vertices are only a subset of the faces, we remove the useless vertices and give the new faces
    idx_Verts_used = np.unique(Faces)
    Verts_used = Verts[idx_Verts_used]
    idx_mapping = np.arange(len(Verts_used))
    mapping = dict(zip(idx_Verts_used,idx_mapping))
    def func(x): 
        return([mapping[x[0]],mapping[x[1]],mapping[x[2]]])
    New_Faces = np.array(list(map(func,Faces)))
    return(Verts_used,New_Faces)




def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)   

@dataclass
class Vertex:
    """Vertex in 2D"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    key: int=0
    on_trijunction = False



@dataclass
class HalfEdge:
    """Half-Edge of a DCEL graph"""

    origin: Vertex = None
    destination: Vertex = None
    material_1: int = 0
    material_2: int = 0
    twin = None
    incident_face: "Face" = None
    prev: "HalfEdge" = None
    next: "HalfEdge" = None
    attached: dict = field(default_factory=dict)
    key: int=0

    def compute_length(self): 
        v = np.zeros(3)
        v[0]=self.origin.x-self.destination.x
        v[1]=self.origin.y-self.destination.y
        v[2]=self.origin.z-self.destination.z
        self.length = np.linalg.norm(v)

    def set_face(self, face):
        if self.incident_face is not None:
            print("Error : the half-edge already has a face.")
            return
        self.incident_face = face
        if self.incident_face.outer_component is None:
            face.outer_component = self

    def set_prev(self, other):
        if other.incident_face is not self.incident_face:
            print("Error setting prev relation : edges must share the same face.")
            return
        self.prev = other
        other.next = self

    def set_next(self, other):
        if other.incident_face is not self.incident_face:
            print("Error setting next relation : edges must share the same face.")
            return
        self.next = other
        other.prev = self

    def set_twin(self, other):
        self.twin = other
        other.twin = other

    def return_vector(self): 
        xo,yo = self.origin.x,self.origin.y
        xt,yt = self.destination.x,self.destination.y
        vect = np.array([xt-xo,yt-yo])
        vect/=np.linalg.norm(vect)
        return(vect)

    def __repr__(self):
        ox = "None"
        oy = "None"
        dx = "None"
        dy = "None"
        if self.origin is not None:
            ox = str(self.origin.x)
            oy = str(self.origin.y)
        if self.destination is not None:
            dx = str(self.destination.x)
            dy = str(self.destination.y)
        return f"origin : ({ox}, {oy}) ; destination : ({dx}, {dy})"


@dataclass
class Face:
    """Face of a DCEL graph"""

    attached: dict = field(default_factory=dict)
    outer_component: HalfEdge = None
    _closed: bool = True
    material_1: int = 0
    material_2: int = 0
    normal = None
    key: int=0

    # def set_outer_component(self, half_edge):
    #     if half_edge.incident_face is not self:
    #         print("Error : the edge must have the same incident face.")
    #         return
    #     self.outer_component = half_edge

    def first_half_edge(self):
        self._closed = False
        first_half_edge = self.outer_component
        if first_half_edge is None:
            return None
        while first_half_edge.prev is not None:
            first_half_edge = first_half_edge.prev
            if first_half_edge is self.outer_component:
                self._closed = True
                break
        return first_half_edge

    def last_half_edge(self):
        self._closed = False
        last_half_edge = self.outer_component
        if last_half_edge is None:
            return None
        while last_half_edge.next is not None:
            last_half_edge = last_half_edge.next
            if last_half_edge is self.outer_component:
                self._closed = True
                last_half_edge = self.outer_component.prev
                break
        return last_half_edge

    def closed(self):
        self.first_half_edge()
        return self._closed

    def get_edges(self):
        edges = []
        if self.outer_component is None:
            return edges

        first_half_edge = self.first_half_edge()
        last_half_edge = self.last_half_edge()
        edge = first_half_edge
        while True:
            edges.append(edge)
            if edge is last_half_edge:
                break
            else:
                edge = edge.next
        return edges

    def get_materials(self): 
        self.material_1 = self.outer_component.material_1
        self.material_2 = self.outer_component.material_2

    def get_vertices(self):
        vertices = []
        for edge in self.get_edges():
            if edge.origin is not None:
                vertices.append(edge.origin)
        return vertices

    def get_area(self):
        if not self.closed():
            return None
        else:

            def distance(p1, p2):
                return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + ((p1.z - p2.z) ** 2))

            area = 0
            vertices = self.get_vertices()
            p1 = vertices[0]
            for i in range(1, len(vertices) - 1):
                p2 = vertices[i]
                p3 = vertices[i + 1]
                a = distance(p1, p2)
                b = distance(p2, p3)
                c = distance(p3, p1)
                s = (a + b + c) / 2.0
                area += math.sqrt(s * (s - a) * (s - b) * (s - c))
            return area

def separate_faces_dict_keep_idx(Faces,n_towers=10): 
    n_towers = np.amax(Faces[:,[3,4]])+1
   
    Occupancy=np.zeros(n_towers)
    Dict={}
    Dict_idx={}
    for idx,face in enumerate(Faces) : 
        _,_,_,num1,num2 = face
        if num1!=-1:
            if Occupancy[num1]==0:
                Dict[num1]=[face[[0,1,2]]]
                Dict_idx[num1]=[idx]
                Occupancy[num1]+=1
            else : 
                Dict[num1].append(face[[0,1,2]])
                Dict_idx[num1].append(idx)
            
        if num2!=-1:
            if Occupancy[num2]==0:
                Dict[num2]=[face[[0,1,2]]]
                Dict_idx[num2]=[idx]
                Occupancy[num2]+=1
            else : 
                Dict[num2].append(face[[0,1,2]])
                Dict_idx[num2].append(idx)
            
            
    Faces_separated={}
    Idx_faces = {}
    for i in sorted(Dict.keys()) : 
        Faces_separated[i] = (np.array(Dict[i]))
        Idx_faces[i] = np.array(Dict_idx[i])
    return(Faces_separated,Idx_faces)

def compute_scattered_arrays(Mesh,coeff):
    Verts,Faces = Mesh.v, Mesh.f
        
    Clusters,Clusters_idx = separate_faces_dict_keep_idx(Faces)
    maxkey = np.amax(Faces[:,3:])
    All_verts=[]
    All_faces=[]
    All_idx=[]
    offset = 0 
    embryo_centroid = np.mean(Verts,axis=0)
    clusters_displacements = {}

    for key in sorted(list(Clusters.keys())) : 
        if key ==0 : 
            continue
        faces = np.array(Clusters[key])

        vn,fn = renormalize_verts(Verts,faces)
        
        #to change with a formula from ddg
        array_centroid = np.mean(vn,axis=0)
        vn = vn + coeff * (array_centroid - embryo_centroid)
        clusters_displacements[key]=(coeff * (array_centroid - embryo_centroid)).copy()

        All_verts.append(vn.copy())
        All_faces.append(fn.copy()+offset)
        All_idx.append(Clusters_idx[key])
        
        offset+=len(vn)
    All_verts = np.vstack(All_verts)
    All_faces = np.vstack(All_faces)
    All_idx = np.hstack(All_idx)
    Mesh.v_scattered = All_verts
    Mesh.f_scattered = All_faces
    Mesh.idx_scattered = All_idx
    Mesh.clusters_displacements = clusters_displacements
        

class DCEL_Data:
    """DCEL Graph containing faces, half-edges and vertices."""
    #Take a multimaterial mesh as an entry
    def __init__(self,Verts,Faces):
        for i, f in enumerate(Faces): 
            if f[3]>f[4]: 
                Faces[i]=Faces[i,[0,2,1,4,3]]
        Verts,Faces = remove_unused_vertices(Verts,Faces)
        self.v = Verts
        self.f = Faces
        #self.n_materials = np.amax(Faces[:,[3,4]])+1
        self.materials = np.unique(Faces[:,[3,4]])
        self.n_materials = len(self.materials)
        Vertices_list, Halfedges_list, Faces_list = build_lists(Verts,Faces)
        self.vertices = Vertices_list
        self.faces = Faces_list
        self.half_edges = Halfedges_list
        self.compute_areas_faces()
        self.compute_centroids_cells()
        self.mark_trijunctional_vertices()
        self.compute_length_halfedges()

    def compute_scattered_arrays(self,coeff):
        compute_scattered_arrays(self,coeff)
        
    def compute_length_halfedges(self): 
        compute_length_halfedges(self)

    def compute_areas_faces(self):
        compute_areas_faces(self)

    def compute_vertex_normals(self): 
        return(compute_vertex_normals(self.v,self.f))

    def compute_verts_faces_interfaces(self):
        return(compute_verts_and_faces_interfaces(self))

    def compute_networkx_graph(self): 
        return(compute_networkx_graph(self))
    
    def find_trijunctional_edges(self):
        return(find_trijunctional_edges(self))

    def compute_centroids_cells(self): 
        self.centroids = {}
        separated_faces = separate_faces_dict(self.f)
        for i in separated_faces.keys():
            self.centroids[i]=np.mean(self.v[np.unique(separated_faces[i]).astype(int)],axis=0)
        
    def mark_trijunctional_vertices(self,return_list=False): 
        return(mark_trijunctional_vertices(self,return_list))
        
    def compute_length_trijunctions(self,prints=False): 
        return(compute_length_trijunctions(self,prints=False))

    #TODO:  FIND AN EFFICIENT IMPLEMENTATION OF THE TRIJUNCTIONAL LENGTH DERIVATIVES
    
    def compute_areas_cells(self):
        return(compute_areas_cells(self))

    def compute_areas_interfaces(self): 
        return(compute_areas_interfaces(self))
    
    def compute_area_derivatives(self): 
        return(compute_area_derivative_dict(self))

    def compute_volumes_cells(self): 
        return(compute_volume_cells(self))
    
    def compute_volume_derivatives(self):
        return(compute_volume_derivative_dict(self))

    def compute_angles_tri(self,unique=True):
        return(compute_angles_tri(self,unique=unique))

    def compute_curvatures_interfaces(self,laplacian="robust",weighted=True):
        #"robust" or "cotan"
        return(compute_curvature_interfaces(self,laplacian=laplacian,weighted=weighted))


    def save(self, filename):
        with open(filename, "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.half_edges, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            self.vertices = pickle.load(f)
            self.half_edges = pickle.load(f)
            self.faces = pickle.load(f)
            



"""
DCEL BUILDING FUNCTIONS
"""
def compute_normal_Faces(Verts,Faces):
    Pos = Verts[Faces[:,[0,1,2]]]
    Sides_1 = Pos[:,1]-Pos[:,0]
    Sides_2 = Pos[:,2]-Pos[:,1]
    Normal_faces = np.cross(Sides_1,Sides_2,axis=1)
    Norms = np.linalg.norm(Normal_faces,axis=1)#*(1+1e-8)
    Normal_faces/=(np.array([Norms]*3).transpose())
    return(Normal_faces)

def build_lists(Verts, Faces):
    Normals = compute_normal_Faces(Verts,Faces)
    Vertices_list = make_vertices_list(Verts)
    Halfedge_list = []
    for i in range(len(Faces)): 
        a,b,c,_,_ = Faces[i]
        Halfedge_list.append(HalfEdge(origin = Vertices_list[a], destination= Vertices_list[b],key=3*i+0))
        Halfedge_list.append(HalfEdge(origin = Vertices_list[c], destination= Vertices_list[a],key=3*i+1))
        Halfedge_list.append(HalfEdge(origin = Vertices_list[b], destination= Vertices_list[c],key=3*i+2))
        
    index=0
    for i in range(len(Faces)): 
        Halfedge_list[index].next = Halfedge_list[index+1]
        Halfedge_list[index].prev = Halfedge_list[index+2]
        
        Halfedge_list[index+1].next = Halfedge_list[index+2]
        Halfedge_list[index+1].prev = Halfedge_list[index]
        
        Halfedge_list[index+2].next = Halfedge_list[index]
        Halfedge_list[index+2].prev = Halfedge_list[index+1]
        
        index+=3
        
    Faces_list = []
    for i in range(len(Faces)): 
        Faces_list.append(Face(outer_component=Halfedge_list[i+3] ,material_1 = Faces[i,3], material_2 = Faces[i,4],key=i))
        Faces_list[i].normal = Normals[i]

    for i in range(len(Faces)): 
        Halfedge_list[3*i+0].incident_face = Faces_list[i]
        Halfedge_list[3*i+1].incident_face = Faces_list[i]
        Halfedge_list[3*i+2].incident_face = Faces_list[i]
        
    #find twins
    F = Faces.copy()[:,[0,1,2]]
    E = np.hstack((F,F)).reshape(-1,2)
    E = np.sort(E,axis=1)
    key_mult = find_key_multiplier(np.amax(F))
    Keys = E[:,1]*key_mult + E[:,0]
    Dict_twins = {}
    for i,key in enumerate(Keys) : 
        Dict_twins[key] = Dict_twins.get(key,[])+[i]
    List_twins = []
    counts = np.zeros(4)
    for i in range(len(E)):
        key = Keys[i]
        l = Dict_twins[key].copy()
        l.remove(i)
        List_twins.append(l)
        
    for i,list_twin in enumerate(List_twins):
        Halfedge_list[i].twin = list_twin

    return(Vertices_list,Halfedge_list,Faces_list)

def make_vertices_list(Verts): 
    Vertices_list = []
    for i,vertex_coords in enumerate(Verts) : 
        x,y,z = vertex_coords
        Vertices_list.append(Vertex(x=x,y=y,z=z,key=i))
    return(Vertices_list)


def mark_trijunctional_vertices(Mesh,return_list = False): 
    list_trijunctional_vertices = []
    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 : 
            Mesh.vertices[edge.origin.key].on_trijunction = True
            Mesh.vertices[edge.destination.key].on_trijunction = True
            list_trijunctional_vertices.append(edge.origin.key)
            list_trijunctional_vertices.append(edge.destination.key)
    if return_list : 
        return(np.unique(list_trijunctional_vertices))


"""
DCEL Geometry functions
"""

def find_trijunctional_edges(Mesh):
    F = Mesh.f
    E = np.vstack((F[:,[0,1]],F[:,[0,2]],F[:,[1,2]]))
    E = np.sort(E,axis=1)
    key_mult = find_key_multiplier(len(Mesh.v)+1)
    K = (E[:,0]+1) + (E[:,1]+1)*key_mult
    Array,Index_first_occurence,Index_inverse,Index_counts = np.unique(K, return_index=True, return_inverse=True, return_counts=True)
    print("Number of trijunctional edges :",np.sum(Index_counts==3))
    Edges_trijunctions = E[Index_first_occurence[Index_counts==3]]
    
    #Verts_concerned = np.unique(Edges_trijunctions)
    return(Edges_trijunctions)

def compute_length_halfedges(Mesh):
    for edge in Mesh.half_edges :
        edge.compute_length()

def compute_faces_areas(Verts,Faces):
    Pos = Verts[Faces[:,[0,1,2]]]
    Sides = Pos-Pos[:,[2,0,1]]
    Lengths_sides = np.linalg.norm(Sides,axis = 2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2

    Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(Areas)

def compute_vertex_normals(Verts,Faces): 
    faces_on_verts = [[] for x in range(len(Verts))]
    for i,f in enumerate(Faces) : 
        faces_on_verts[f[0]].append(i)
        faces_on_verts[f[1]].append(i)
        faces_on_verts[f[2]].append(i)

    Sides = Verts[Faces[:,[0,1,2]]]
    Side_1 = Sides[:,0]-Sides[:,1]
    Side_2 = Sides[:,0]-Sides[:,2]
    Faces_normals = np.cross(Side_1,Side_2,axis=1)
    norms = np.linalg.norm(Faces_normals, axis=1)
    Faces_normals*=np.array([1/norms]*3).transpose()
    Faces_areas = compute_faces_areas(Verts,Faces)
    vertex_normals = np.zeros(Verts.shape)

    for i,f_list in enumerate(faces_on_verts) : 
        c=0
        n=0
        for f_idx in f_list : 
            n+=Faces_normals[f_idx]*Faces_areas[f_idx]
            c+=Faces_areas[f_idx]
        n/=c
        vertex_normals[i]=n
    return(vertex_normals)


def remove_unused_vertices(V,F):
    #Some unused vertices appears after the tetrahedral remeshing. We need to remove them. 
    Verts = V.copy()
    Faces = F.copy()
    faces_on_verts = [[] for x in range(len(Verts))]
    for i,f in enumerate(Faces) : 
        faces_on_verts[f[0]].append(i)
        faces_on_verts[f[1]].append(i)
        faces_on_verts[f[2]].append(i)

    verts_to_remove = []
    for i,f_list in enumerate((faces_on_verts)):
        if len(f_list)==0 : 
            verts_to_remove.append(i)
    
    #print(len(verts_to_remove))
    
    list_verts = np.delete(np.arange(len(Verts)),verts_to_remove)
    idx_new_verts = np.arange(len(list_verts))
    mapping = dict(zip(list_verts,idx_new_verts))

    Verts = Verts[list_verts]
    for i in range(len(Faces)): 
        Faces[i,0]=mapping[Faces[i,0]]
        Faces[i,1]=mapping[Faces[i,1]]
        Faces[i,2]=mapping[Faces[i,2]]
    return(Verts,Faces)



def compute_centroids_graph(Mesh): 
    Centroids = np.zeros((Mesh.n_materials,3))
    Faces_dict = separate_faces_dict(Mesh.f)
    for i in Faces_dict.keys():
        Centroids[i]=np.mean(Mesh.v[Faces_dict[i]].reshape(-1,3),axis=0) 
    return(Centroids)

def compute_networkx_graph(Mesh):
    Verts_interfaces,Faces_interfaces=Mesh.compute_verts_faces_interfaces()
    Areas = Mesh.compute_areas_cells()
    Volumes = Mesh.compute_volumes_cells()
    Areas_interfaces = Mesh.compute_areas_interfaces()
    Curvatures = Mesh.compute_curvatures_interfaces()
    

    #Mesh.compute_centroids_cells()

    Centroids = Mesh.centroids
    #Centroids = compute_centroids_graph(Mesh)
    
    G = networkx.Graph()
    Dicts = [{'area':Areas[x],'volume':Volumes[x],'centroid':Centroids[x]} for x in Mesh.materials]
    G.add_nodes_from(zip(Mesh.materials,Dicts))

    edges_array = [(tup[0],tup[1],{'mean_curvature':Curvatures[tup],'area':Areas_interfaces[tup],'verts':Verts_interfaces[tup],'faces':Faces_interfaces[tup]}) for tup in Curvatures.keys()]
    G.add_edges_from(edges_array)
    return(G)

def update_graph_with_scattered_values(G,Mesh): 
    new_centroids = dict(G.nodes.data('centroid'))
    for key in new_centroids : 
        if key==0 : continue
        new_centroids[key]+=Mesh.clusters_displacements[key]
        #print(new_centroids[key])
    networkx.set_node_attributes(G, new_centroids, "centroid")

    verts_faces_dict = {}
    for elmt in G.edges.data('verts'): 
        a,b, v = elmt
        if a==0 : 
            verts_faces_dict[(a,b)] = v.copy() + Mesh.clusters_displacements[b]
        else : 
            verts_faces_dict[(a,b)]= v.copy()

    networkx.set_edge_attributes(G, verts_faces_dict, "verts")
    
    return(G)
    
    

def compute_verts_and_faces_interfaces(Mesh) :
    key_mult = np.amax(Mesh.f[:,[3,4]])+1
    keys,inv_1 = np.unique(Mesh.f[:,3]+Mesh.f[:,4]*key_mult,return_inverse=True)

    interfaces = [(key%key_mult,key//key_mult) for key in keys]
    Faces_dict = {interfaces[i]:Mesh.f[:,:3][keys[inv_1]==keys[i]] for i in range(len(keys))}
    Faces_interfaces={}
    Verts_interfaces={}

    for key in Faces_dict.keys():
        v,f = renormalize_verts(Mesh.v,Faces_dict[key])
        Faces_interfaces[key]=f.copy()
        Verts_interfaces[key]=v.copy()
    
    return(Verts_interfaces,Faces_interfaces)