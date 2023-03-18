
#from Mesh_centering import center_verts
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import skimage.io as io
from skimage.segmentation import watershed
import trimesh
from trimesh import remesh
import numpy as np 

def create_coords(nx,ny,nz):
    XV = np.linspace(0,1,nx)
    YV = np.linspace(0,1,ny)
    ZV = np.linspace(0,1,nz)
    xvv, yvv, zvv = np.meshgrid(XV,YV,ZV)
    xvv=np.transpose(xvv,(1,0,2)).flatten()
    yvv=np.transpose(yvv,(1,0,2)).flatten()
    zvv=zvv.flatten()
    Points=np.vstack(([xvv,yvv,zvv])).transpose()
    return(Points)


def create_mesh_semantic_masks(Verts_0,Faces_0,grid_size):
    Verts,Faces = Verts_0.copy()[:,[2,1,0]],Faces_0[:,[0,1,2]].copy()
    for i in range(3): 
        Verts[:,i]/=grid_size[[2,1,0]][i]
        
    dmax = 1/np.amax(grid_size)
    #print("start of the subidivision")
    Verts, Faces = subdivide_mesh(Verts,Faces,max_edge = dmax/2)
    #print("subdivision finished")
    membrane = make_mask(Verts,grid_size)
    return(membrane)

def subdivide_mesh(Verts,Faces,max_edge): 

    mesh = trimesh.Trimesh(vertices=Verts,
                           faces=Faces)


    longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                  mesh.vertices[mesh.edges[:, 1]],
                                  axis=1).max()
    max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)*2
    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    Verts,Faces = remesh.subdivide_to_size(mesh.vertices,
                                    mesh.faces,
                                    max_edge=max_edge,
                                    max_iter=max_iter)
    return(Verts,Faces)
def make_mask(Verts,grid_size):
    nx,ny,nz = grid_size[[2,1,0]]
    #nx,ny,nz = grid_size
    Points = create_coords(nx,ny,nz)
    Tree = cKDTree(Points)
    Distances = Tree.query(Verts)
    
    dist,idx = Distances
    membrane = np.zeros(nx*ny*nz)
    membrane[idx]=1
    membrane=membrane.reshape(nx,ny,nz)
    return(membrane)

def create_mesh_instance_masks(Verts, Faces, image_shape, seeds):
    semantic_mask = create_mesh_semantic_masks(Verts,Faces, image_shape).transpose(2,1,0)
    distance = ndi.distance_transform_edt(1-semantic_mask)
    markers = np.zeros(distance.shape)
    for i in range(len(seeds)): 
        markers[tuple(seeds[i].T)]=i+1
    labels = watershed(-distance, markers)
    #labels=labels[::-1]
    return(labels)

def reconstruct_mask_from_dict(filename_dict): 
    Dict_mask = np.load(filename_dict,allow_pickle = True).item()
    Verts = Dict_mask["Verts"]
    Faces = Dict_mask["Faces"]
    seeds = Dict_mask["seeds"]
    image_shape = Dict_mask["image_shape"]
    labels = create_mesh_instance_masks(Verts,Faces, image_shape,seeds) -1
    return(labels)