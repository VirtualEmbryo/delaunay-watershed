from scipy import ndimage
from skimage.feature import peak_local_max
import numpy as np
from scipy.spatial import Delaunay
from skimage.segmentation import find_boundaries

from dw2d.deterministic_triangulation import TriangulationOfPoints, deterministic_triangles


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



def pad_mask(mask,pad_size = 1): 
    padded_mask = mask.copy()[pad_size:-pad_size,pad_size:-pad_size]
    padded_mask = np.pad(padded_mask, ((pad_size, pad_size), (pad_size, pad_size)), 'constant',constant_values = 1)
    return(padded_mask)
def give_corners(img): 
    Points=np.zeros((4,2))
    index=0
    a,b = img.shape
    for i in [0,a-1]: 
        for j in [0,b-1]: 
            Points[index]=np.array([i,j])
            index+=1
    return(Points)

def build_triangulation(labels,min_distance=4,deterministic_triangulation=False):
    """Seeds, labels, Delaunay triangulation of the EDT extrema and image corners, and the EDT.

    ``deterministic_triangulation`` (default False here; ``geometry_reconstruction_2d`` passes True on its
    default route): re-triangulate the
    cocircular cells of Qhull's triangulation by a fixed rule and list the triangles in canonical
    order, so the triangulation is the same on every machine
    (``dw2d.deterministic_triangulation.deterministic_triangles``).
    """
    b = StandardLabelToBoundary()(labels.reshape(1,labels.shape[0],labels.shape[1]))[0,0]
    mask_2 = b
    EDT_2 = ndimage.distance_transform_edt(mask_2)
    b = pad_mask(b)
    mask_1 = 1-b
    EDT_1 = ndimage.distance_transform_edt(mask_1)
    inv = np.amax(EDT_2)-EDT_2
    Total_EDT = (EDT_1+np.amax(EDT_2))*mask_1 + inv*mask_2

    seeds_coords = []

    values_lbls = np.unique(labels) 
    for i in values_lbls:
        seed = np.argmax(Total_EDT*((labels==i).astype(float)))
        seeds_coords.append([seed//labels.shape[1],seed%labels.shape[1]])

    seeds_coords = np.array(seeds_coords)
    seeds_indices = values_lbls

    points = peak_local_max(-Total_EDT,min_distance=min_distance,exclude_border=False)
    local_maxes = peak_local_max(Total_EDT,min_distance=min_distance,exclude_border=False)
    corners = give_corners(Total_EDT)

    all_points = np.vstack((points,corners,local_maxes))
    tesselation=Delaunay(all_points)
    if deterministic_triangulation:
        tesselation = TriangulationOfPoints(tesselation.points, deterministic_triangles(tesselation.points, tesselation.simplices))
    
    return(seeds_coords,seeds_indices, tesselation,Total_EDT)

