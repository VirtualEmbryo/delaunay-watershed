#Sacha Ichbiah, Sept 2021
import numpy as np 
import scipy.sparse as sp
import robust_laplacian
import trimesh

#TODO: IMPLEMENT THE EDGE_BASED CURVATURE FORMULA

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)   

def compute_laplacian_cotan(Mesh): 
    ### Traditional cotan laplacian : from
    # from "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds", Meyer et al. 2003

    #Implementation and following explanation from : pytorch3d/loss/mesh_laplacian_smoothing.py

    r"""
    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN matrix such that LV gives a matrix of vectors:
    LV[i] gives the normal scaled by the discrete mean curvature. 
    For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.
    .. code-block:: python
               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij
        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.
    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.
    .. code-block:: python
               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C
        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have
        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a
        Putting these together, we get:
        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH
    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.
    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, inv_areas=laplacian_cot(verts,faces)
    inv_areas = inv_areas.reshape(-1)
    sum_cols = np.array(L.sum(axis=1))
    Laplacian = L@verts - verts*sum_cols
    norm = (0.75*inv_areas).reshape(-1,1)
    return(Laplacian*norm,inv_areas)

def compute_laplacian_robust(Mesh): 
    ### Robust Laplacian using implicit triangulations : 
    # from "A Laplacian for Nonmanifold Triangle Meshes", N.Sharp, K.Crane, 2020
    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, M=robust_laplacian.mesh_laplacian(Mesh.v,Mesh.f[:,[0,1,2]])
    inv_areas = 1/M.diagonal().reshape(-1)/3
    Sum_cols = np.array(L.sum(axis=1)) #Useless as it is already 0 (sum comprised in the central term) see http://rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes
    first_term = np.dot(L.toarray(),verts)
    second_term = verts*Sum_cols
    Laplacian = (first_term-second_term)
    norm = (1.5*inv_areas).reshape(-1,1)
    return(-Laplacian*norm,inv_areas)

def compute_gaussian_curvature_vertices(Mesh):
    mesh_trimesh =trimesh.Trimesh(vertices=Mesh.v,
                  faces = Mesh.f[:,:3]) 
    G = trimesh.curvature.discrete_gaussian_curvature_measure(mesh_trimesh, Mesh.v,0.)
    return(G)

    
def compute_curvature_vertices_cotan(Mesh): 
    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, inv_areas=laplacian_cot(Mesh.v,Mesh.f[:,[0,1,2]])
    inv_areas = inv_areas.reshape(-1)
    Sum_cols = np.array(L.sum(axis=1))
    first_term = np.dot(L.toarray(),verts)
    second_term = verts*Sum_cols
    Laplacian = (first_term-second_term)/2
    H = np.linalg.norm(Laplacian,axis=1)*3*inv_areas/2
    return(H,inv_areas,Laplacian*3*(np.array([inv_areas]*3).transpose())/2)

def compute_curvature_vertices_robust_laplacian(Mesh): 
    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, M=robust_laplacian.mesh_laplacian(Mesh.v,Mesh.f[:,[0,1,2]])
    inv_areas = 1/M.diagonal().reshape(-1)/3
    Sum_cols = np.array(L.sum(axis=1))
    first_term = np.dot(L.toarray(),verts)
    second_term = verts*Sum_cols
    Laplacian = (first_term-second_term)
    H = np.linalg.norm(Laplacian,axis=1)*3*inv_areas/2
    return(H,inv_areas,Laplacian*3*(np.array([inv_areas]*3).transpose())/2)


def compute_curvature_interfaces(Mesh,laplacian = "robust",weighted=True): 
    Interfaces={}
    Interfaces_weights={}
    if laplacian =="robust" : 
        L,inv_areas = compute_laplacian_robust(Mesh)
    elif laplacian =="cotan" : 
        L,inv_areas = compute_laplacian_cotan(Mesh)

    vertex_normals = Mesh.compute_vertex_normals()
    H = np.sign(np.sum(np.multiply(L,vertex_normals),axis=1))*np.linalg.norm(L,axis=1)
    
    
    Vertices_on_interfaces ={}
    for edge in Mesh.half_edges : 
        #pass trijunctions
        
        materials = (edge.incident_face.material_1,edge.incident_face.material_2)
        interface_key = (min(materials),max(materials))
        
        Vertices_on_interfaces[interface_key]=Vertices_on_interfaces.get(interface_key,[])
        Vertices_on_interfaces[interface_key].append(edge.origin.key)
        Vertices_on_interfaces[interface_key].append(edge.destination.key)
        
    verts_idx = {}
    for key in Vertices_on_interfaces.keys() : 
        verts_idx[key] = np.unique(np.array(Vertices_on_interfaces[key]))
        
    Interfaces_curvatures = {}
    for key in verts_idx.keys() : 
        curvature = 0
        weights = 0
        for vert_idx in verts_idx[key]: 
            v = Mesh.vertices[vert_idx]
            if v.on_trijunction : 
                continue
            else : 
                if weighted : 
                    curvature += H[vert_idx]/inv_areas[vert_idx]
                    weights += 1/inv_areas[vert_idx]
                else : 
                    curvature += H[vert_idx]
                    weights +=1

        """
        TEMPORARY : 
        FOR THE MOMENT, WE CANNOT COMPUTE CURVATURE ON LITTLE INTERFACES
        THERE ARE THREE POSSIBILITIES : 
        -WE REFINE THE SURFACES UNTIL THERE IS A VERTEX ON THE SURFACE AND THUS IT CAN BE COMPUTED> 
        -WE PUT THE CURVATURE TO ZERO 
        -WE REMOVE THE EQUATION FROM THE SET OF EQUATIONS

        -Removing the equations could be dangerous as we do not know what we are going to get : 
        maybe the system will become underdetermined, and thus unstable ? 
            -> Unprobable as the systems are strongly overdetermined. 
            -> Bayesian ? 
        -Putting the curvature to zero should not have a strong influence on the inference as in any way during the least-squares
        minimization each equation for the pressures are proportionnal to the area of the interfaces. 
        Thus we return to the case 1, where in fact it does not matter so much if the equation is removed or kept for the little interfaces
        
        -It is thus useless to refine the surface until a curvature can be computed, for sure. 
        
        
        """
        if weights==0 : 
            Interfaces_curvatures[key]=np.nan
        else : 
            Interfaces_curvatures[key]=curvature/weights
    
    return(Interfaces_curvatures)


def cot(x):
    return(1/np.tan(x))
  


def laplacian_cot(verts,faces):
    ##


    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.
    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        2-element tuple containing
        - **L**: FloatTensor of shape (V,V) for the Laplacian matrix (V = sum(V_n))
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """

    V, F = len(verts),len(faces)

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = np.linalg.norm((v1 - v2),axis=1)
    B = np.linalg.norm((v0 - v2),axis=1)
    C = np.linalg.norm((v0 - v1),axis=1)
    
    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = np.sqrt((s * (s - A) * (s - B) * (s - C)))#.clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = np.stack([cota, cotb, cotc], axis=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = np.stack([ii,jj],axis=0).reshape(2, F*3)
    
    L = sp.coo_matrix((cot.reshape(-1),(idx[1],idx[0])), shape = (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.transpose()

   
    # For each vertex, compute the sum of areas for triangles containing it.
    inv_areas=np.zeros(V)
    idx = faces.reshape(-1)
    val = np.stack([area] * 3, axis=1).reshape(-1)
    np.add.at(inv_areas,idx,val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.reshape(-1, 1)

    return L,inv_areas












#from https://jekel.me/2015/Least-Squares-Sphere-Fit/

def sphereFit(verts):
    #   Assemble the A matrix
    spX = verts[:,0]
    spY = verts[:,1]
    spZ = verts[:,2]
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)
    print(residules/len(verts))
    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, np.array([C[0], C[1], C[2]])[:,0]
def sphereFit_residue(verts):
    #   Assemble the A matrix
    spX = verts[:,0]
    spY = verts[:,1]
    spZ = verts[:,2]
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)
    if len(residules)>0: 
        return(residules[0]/len(verts))
    else: return(0)
    
def compute_sphere_fit_residues_dict(G):
    Sphere_fit_residues_faces = {}
    for key in G.edges.keys():
        vn = G.edges[key]['verts']
        Sphere_fit_residues_faces[key] = sphereFit_residue(vn)
    return(Sphere_fit_residues_faces)



def compute_areas_interfaces(Mesh): 
    ###
    #Duplicate of a function present in Geometry (with the same name), but computed in a different manner
    ###

    f = Mesh.f
    v = Mesh.v
    Areas = compute_areas(f[:,[0,1,2]],v)
    Interfaces_areas = {}
    for i, face in enumerate(f): 
        _,_,_,a,b = face
        Table = Interfaces_areas.get((a,b),0)
        Interfaces_areas[(a,b)]=Table+Areas[i]
    return(Interfaces_areas)

def compute_areas(faces,verts): 
    Pos = verts[faces]
    Sides = Pos-Pos[:,[2,0,1]]
    Lengths_sides = np.linalg.norm(Sides,axis = 2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2

    Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(Areas)




