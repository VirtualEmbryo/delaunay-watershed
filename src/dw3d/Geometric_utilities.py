from skimage.feature import peak_local_max
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
from edt import edt as euclidean_dt
from time import time
import torch
from skimage.segmentation import find_boundaries


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, mode='thick', foreground=False,
                 **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype('int32')

        results = []
        if self.foreground:
            foreground = (m > 0).astype('uint8')
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)

def interpolate_image(image):
    x = np.linspace(0,image.shape[0]-1,image.shape[0])
    y = np.linspace(0,image.shape[1]-1,image.shape[1])
    z = np.linspace(0,image.shape[2]-1,image.shape[2])
    image_interp = RegularGridInterpolator((x,y,z) ,image)
    return(image_interp)


def pad_mask(mask,pad_size = 1): 
    padded_mask = mask.copy()[pad_size:-pad_size,pad_size:-pad_size,pad_size:-pad_size]
    padded_mask = np.pad(padded_mask, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), 'constant',constant_values = 1)
    return(padded_mask)

def give_corners(img): 
    Points=np.zeros((8,3))
    index=0
    a,b,c = img.shape
    for i in [0,a-1]: 
        for j in [0,b-1]: 
            for k in [0,c-1]:
                Points[index]=np.array([i,j,k])
                index+=1
    return(Points)

def create_coords(nx,ny,nz):
    XV = np.linspace(0,nx-1,nx)
    YV = np.linspace(0,ny-1,ny)
    ZV = np.linspace(0,nz-1,nz)
    xvv, yvv, zvv = np.meshgrid(XV,YV,ZV)
    xvv=np.transpose(xvv,(1,0,2)).flatten()
    yvv=np.transpose(yvv,(1,0,2)).flatten()
    zvv=zvv.flatten()
    Points=np.vstack(([xvv,yvv,zvv])).transpose().astype(int)
    return(Points)












def build_triangulation(labels,min_distance=5,mode='torch',prints=False):
    if mode == 'torch': 
        return(build_triangulation_torch(labels,(min_distance//2)*2 +1,prints))
    else : 
        return(build_triangulation_skimage(labels,min_distance,prints))

def build_triangulation_torch(labels,min_distance=5,prints=False):#,size_shell=2,dist_shell=4):
    if prints : 
        print("Mode == Torch")
        print("Kernel size =",min_distance)
        print("Computing EDT ...")
    t1 = time()
    b = StandardLabelToBoundary()(labels)[0]
    mask_2 = b
    EDT_2 = euclidean_dt(mask_2)
    #b = pad_mask(b)
    mask_1 = 1-b
    EDT_1 = euclidean_dt(mask_1)
    inv = np.amax(EDT_2)-EDT_2
    Total_EDT = (EDT_1+np.amax(EDT_2))*mask_1 + inv*mask_2
    t2 = time()
    if prints : print("EDT computed in ",np.round(t2-t1,2))

    seeds_coords = []

    nx,ny,nz = labels.shape
    table_coords = create_coords(nx,ny,nz)
    values_lbls = np.unique(labels) 
        
    flat_edt = Total_EDT.flatten()
    flat_labels = labels.flatten()
    for i in values_lbls:
        f_i = flat_labels==i
        seed = np.argmax(flat_edt[f_i])
        seeds_coords.append(table_coords[f_i][seed])
        
    seeds_coords = np.array(seeds_coords)
    seeds_indices = values_lbls

    corners = give_corners(Total_EDT)
    
    
    t3 = time()
    if prints : print("Searching local extremas ...")
    EDT =Total_EDT + np.random.rand(nx,ny,nz)*1e-5
    T = torch.tensor(EDT).unsqueeze(0)
    kernel_size = min_distance
    padding = kernel_size//2
    F = torch.nn.MaxPool3d((kernel_size,kernel_size,kernel_size), stride=(1,1,1), padding=padding, dilation=1, return_indices=False, ceil_mode=False)

    minpooled = F(-T)[0].numpy()
    markers_min = (EDT+minpooled)==0
    local_mins = table_coords[markers_min.flatten()]
    if prints : print("Number of local minimas :",len(local_mins))

    maxpooled = F(T)[0].numpy()
    markers_max = (EDT-maxpooled)==0
    local_maxes = table_coords[markers_max.flatten()]
    if prints : print("Number of local maxes :",len(local_maxes))
    
    t4 = time()
    if prints : print("Local minimas computed in ",np.round(t4-t3,2))
    
    all_points = np.vstack((corners,local_maxes,local_mins))
    
    if prints : print("Starting triangulation..")
    
    tesselation=Delaunay(all_points)

    t5 = time()
    if prints : print("Triangulation build in ",np.round(t5-t4,2))
    
    return(seeds_coords,seeds_indices, tesselation,Total_EDT)



def build_triangulation_skimage(labels,min_distance=5,prints=False):
    if prints : 
        print("Mode == Skimage")
        print("min_distance =",min_distance)
        print("Computing EDT ...")
    t1 = time()

    b = StandardLabelToBoundary()(labels)[0]
    mask_2 = b
    EDT_2 = euclidean_dt(mask_2)
    b = pad_mask(b)
    mask_1 = 1-b
    EDT_1 = euclidean_dt(mask_1)
    inv = np.amax(EDT_2)-EDT_2
    Total_EDT = (EDT_1+np.amax(EDT_2))*mask_1 + inv*mask_2
    t2 = time()
    if prints : print("EDT computed in ",np.round(t2-t1,2))

    seeds_coords = []

    nx,ny,nz = labels.shape
    table_coords = create_coords(nx,ny,nz)
    values_lbls = np.unique(labels) 
        
    flat_edt = Total_EDT.flatten()
    flat_labels = labels.flatten()
    for i in values_lbls:
        f_i = flat_labels==i
        seed = np.argmax(flat_edt[f_i])
        seeds_coords.append(table_coords[f_i][seed])
        
    seeds_coords = np.array(seeds_coords)
    seeds_indices = values_lbls

    corners = give_corners(Total_EDT)
    
    
    t3 = time()
    if prints : print("Searching local extremas ...")
    EDT = Total_EDT + np.random.rand(nx,ny,nz)*1e-5
    
    local_mins = peak_local_max(-EDT,min_distance=min_distance,exclude_border=False)
    if prints : print("Number of local minimas :",len(local_mins))

    local_maxes = peak_local_max(EDT,min_distance=min_distance,exclude_border=False)
    if prints : print("Number of local maxes :",len(local_maxes))
    
    t4 = time()
    if prints : print("Local minimas computed in ",np.round(t4-t3,2))
    
    all_points = np.vstack((corners,local_maxes,local_mins))
    
    if prints : print("Starting triangulation..")
    
    tesselation=Delaunay(all_points)

    t5 = time()
    if prints : print("Triangulation build in ",np.round(t5-t4,2))
    
    return(seeds_coords,seeds_indices, tesselation,Total_EDT)

