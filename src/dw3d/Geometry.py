#Sacha Ichbiah, Sept 2021
import numpy as np 
from tqdm import tqdm
import torch

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)    

## LENGTHS AND DERIVATIVES

def compute_area_derivative_autodiff(Mesh,device = 'cpu'):

    #Faces_membranes = extract_faces_membranes(Mesh)
    key_mult = np.amax(Mesh.f[:,3:])+1
    keys = Mesh.f[:,3]+key_mult*Mesh.f[:,4]
    Faces_membranes = {}
    for key in np.unique(keys):
        tup = (key%key_mult,key//key_mult)
        Faces_membranes[tup]=Mesh.f[:,:3][np.arange(len(keys))[keys==key]]

    verts = torch.tensor(Mesh.v,dtype=torch.float,requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts],lr=1) # Useless, here just to reset the grad

    Areas_derivatives = {}
    for tup in sorted(Faces_membranes.keys()):

        loss_area = (Compute_Area_Faces_torch(verts,torch.tensor(Faces_membranes[tup]))).sum()
        loss_area.backward()
        Areas_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return(Areas_derivatives)




def compute_volume_derivative_autodiff_dict(Mesh,device='cpu'):
    
    #Faces_manifolds = extract_faces_manifolds(Mesh)
    Faces_manifolds = {key:[] for key in Mesh.materials}
    for face in Mesh.f :
        a,b,c,m1,m2 = face
        Faces_manifolds[m1].append([a,b,c])
        Faces_manifolds[m2].append([a,c,b])
    
    verts = torch.tensor(Mesh.v,dtype=torch.float,requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts],lr=1) # Useless, here just to reset the grad

    Volumes_derivatives = {}
    for key in Mesh.materials:#1:] :
        faces = Faces_manifolds[key]
        assert len(faces)>0
        loss_volume = -Compute_Volume_manifold_torch(verts,torch.tensor(faces))
        loss_volume.backward()
        Volumes_derivatives[key]=(verts.grad.numpy().copy())
        optimizer.zero_grad()
    
    return(Volumes_derivatives)

def compute_length_derivative_autodiff(Mesh,device = 'cpu'):

    Edges_trijunctions = extract_edges_trijunctions(Mesh)

    verts = torch.tensor(Mesh.v,dtype=torch.float,requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts],lr=1) # Useless, here just to reset the grad

    Length_derivatives = {}
    for tup in sorted(Edges_trijunctions.keys()):
        loss_length = (Compute_length_edges_trijunctions_torch(verts,torch.tensor(Edges_trijunctions[tup]))).sum()
        loss_length.backward()
        Length_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return(Length_derivatives)
def Compute_length_edges_trijunctions_torch(verts,Edges_trijunctions):
    Pos = verts[Edges_trijunctions]
    Lengths = torch.norm(Pos[:,0]-Pos[:,1],dim=1)
    return(Lengths)



def compute_length_trijunctions(Mesh,prints=False): 
    Length_trijunctions = {}
    Edges_trijunctions = extract_edges_trijunctions(Mesh,prints)
    for key in Edges_trijunctions.keys(): 
        Length_trijunctions[key] = np.sum(Compute_length_edges_trijunctions(Mesh.v,Edges_trijunctions[key]))
    return(Length_trijunctions)

def Compute_length_edges_trijunctions(verts,Edges_trijunctions):
    Pos = verts[Edges_trijunctions]
    Lengths = np.linalg.norm(Pos[:,0]-Pos[:,1],axis=1)
    return(Lengths)
"""
    F = Mesh.f
    E = np.vstack((F[:,[0,1]],F[:,[0,2]],F[:,[1,2]]))
    E = np.sort(E,axis=1)
    Zones = np.vstack((F[:,[3,4]],F[:,[3,4]],F[:,[3,4]]))
    key_mult = find_key_multiplier(len(Mesh.v)+1)
    K = (E[:,0]+1) + (E[:,1]+1)*key_mult
    Array,Index_first_occurence,Index_inverse,Index_counts = np.unique(K, return_index=True, return_inverse=True, return_counts=True)
    if prints : 
        print("Number of trijunctional edges :",np.sum(Index_counts==3))
    Edges_trijunctions = E[Index_first_occurence[Index_counts==3]]
    

    Indices = np.arange(len(Index_counts))
    Map = {key:[] for key in Indices[Index_counts==3]}
    Table = np.zeros(len(Index_counts))
    Table[Index_counts==3]+=1

    for i in range(len(Index_inverse)): 
        inverse = Index_inverse[i]
        if Table[inverse]==1 : 
            Map[inverse].append(i)

    Trijunctional_line={}
    for key in sorted(Map.keys()) : 
        x = Map[key]
        regions = np.hstack((Zones[x[0]],Zones[x[1]],Zones[x[2]]))
        u = np.unique(regions)
        if len(u)>4 : 
            print("oui")
            continue
        else : 
            Trijunctional_line[tuple(u)]=Trijunctional_line.get(tuple(u),[])
            Trijunctional_line[tuple(u)].append(E[x[0]])
            assert(E[x[0]][0]==E[x[1]][0]==E[x[2]][0] and E[x[0]][1]==E[x[1]][1]==E[x[2]][1])
    
    Output_dict = {}
    for key in sorted(Trijunctional_line.keys()):
        Output_dict[key] = np.vstack(Trijunctional_line[key])
    return(Output_dict)

"""
def extract_edges_trijunctions(Mesh,prints=False):
    F = Mesh.f
    E = np.vstack((F[:,[0,1]],F[:,[0,2]],F[:,[1,2]]))
    E = np.sort(E,axis=1)
    Zones = np.vstack((F[:,[3,4]],F[:,[3,4]],F[:,[3,4]]))
    key_mult = find_key_multiplier(len(Mesh.v)+1)
    K = (E[:,0]+1) + (E[:,1]+1)*key_mult
    Array,Index_first_occurence,Index_inverse,Index_counts = np.unique(K, return_index=True, return_inverse=True, return_counts=True)
    if prints : 
        print("Number of trijunctional edges :",np.sum(Index_counts==3))
    Edges_trijunctions = E[Index_first_occurence[Index_counts==3]]
    

    Indices = np.arange(len(Index_counts))
    Map = {key:[] for key in Indices[Index_counts==3]}
    Table = np.zeros(len(Index_counts))
    Table[Index_counts==3]+=1

    for i in range(len(Index_inverse)): 
        inverse = Index_inverse[i]
        if Table[inverse]==1 : 
            Map[inverse].append(i)

    Trijunctional_line={}
    for key in sorted(Map.keys()) : 
        x = Map[key]
        regions = np.hstack((Zones[x[0]],Zones[x[1]],Zones[x[2]]))
        u = np.unique(regions)
        if len(u)>4 : 
            print("oui")
            continue
        else : 
            Trijunctional_line[tuple(u)]=Trijunctional_line.get(tuple(u),[])
            Trijunctional_line[tuple(u)].append(E[x[0]])
            assert(E[x[0]][0]==E[x[1]][0]==E[x[2]][0] and E[x[0]][1]==E[x[1]][1]==E[x[2]][1])
    
    Output_dict = {}
    for key in sorted(Trijunctional_line.keys()):
        Output_dict[key] = np.vstack(Trijunctional_line[key])
    return(Output_dict)

## AREAS AND DERIVATIVES

def Compute_Area_Faces(Verts,Faces):
    Pos = Verts[Faces]
    Sides = Pos-Pos[:,[2,0,1]]

    Lengths_sides =np.norm(Sides,dim=2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2
    Diffs = np.zeros(Lengths_sides.shape)
    Diffs[:,0] = Half_perimeters - Lengths_sides[:,0]
    Diffs[:,1] = Half_perimeters - Lengths_sides[:,1]
    Diffs[:,2] = Half_perimeters - Lengths_sides[:,2]
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(Areas)

def Compute_Area_Faces_torch(Verts,Faces):
    Pos = Verts[Faces]
    Sides = Pos-Pos[:,[2,0,1]]

    Lengths_sides =torch.norm(Sides,dim=2)
    Half_perimeters = torch.sum(Lengths_sides,axis=1)/2
    Diffs = torch.zeros(Lengths_sides.shape)
    Diffs[:,0] = Half_perimeters - Lengths_sides[:,0]
    Diffs[:,1] = Half_perimeters - Lengths_sides[:,1]
    Diffs[:,2] = Half_perimeters - Lengths_sides[:,2]
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(Areas)



def compute_areas_faces(Mesh):
    Pos = Mesh.v[Mesh.f[:,[0,1,2]]]
    Sides = Pos-Pos[:,[2,0,1]]
    Lengths_sides = np.linalg.norm(Sides,axis = 2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2

    Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    for i,face in enumerate(Mesh.faces) : 
        face.area = Areas[i]

def compute_areas_cells(Mesh):
    areas={key:0 for key in Mesh.materials}
    for face in Mesh.faces : 
        areas[face.material_1]+=face.area
        areas[face.material_2]+=face.area
    return(areas)

def compute_areas_interfaces(Mesh):
    Interfaces={}
    for face in Mesh.faces : 
        materials = (face.material_1,face.material_2)
        key = (min(materials),max(materials))
        Interfaces[key]=Interfaces.get(key,0)+face.area
    return(Interfaces)

def compute_area_derivative_dict(Mesh):
    
    T = np.array(sorted(compute_areas_interfaces(Mesh).keys()))
    kt = Mesh.n_materials+1
    Areas = {tuple(t):0 for t in T}
    Verts, Faces, Faces_label = Mesh.v, Mesh.f[:,:3], Mesh.f[:,3:]
    DA = {tuple(t) : np.zeros((len(Verts),3)) for t in T}
    
    Coords = Verts[Faces]
    X,Y,Z = Coords[:,0],Coords[:,1],Coords[:,2]
    e3 = np.cross(Z-Y,X-Z)/(np.linalg.norm(np.cross(Z-Y,X-Z),axis=1).reshape(-1,1))
    cross_e3_x = np.cross(e3,Z-Y)/2
    cross_e3_y = np.cross(e3,X-Z)/2
    cross_e3_z = np.cross(e3,Y-X)/2

    List_indices_faces_per_vertices_x={key:[[] for i in range(len(Verts))] for key in DA.keys()}
    List_indices_faces_per_vertices_y={key:[[] for i in range(len(Verts))] for key in DA.keys()}
    List_indices_faces_per_vertices_z={key:[[] for i in range(len(Verts))] for key in DA.keys()}

    for i, face in enumerate(Faces):
        i_x,i_y,i_z = Faces[i]
        a,b = Faces_label[i]
        List_indices_faces_per_vertices_x[(a,b)][i_x].append(i)
        List_indices_faces_per_vertices_y[(a,b)][i_y].append(i)
        List_indices_faces_per_vertices_z[(a,b)][i_z].append(i)

    for key in tqdm(DA.keys()):
        for iv in range(len(Verts)):
            DA[key][iv] = np.vstack((cross_e3_x[List_indices_faces_per_vertices_x[key][iv]],
                                     cross_e3_y[List_indices_faces_per_vertices_y[key][iv]],
                                     cross_e3_z[List_indices_faces_per_vertices_z[key][iv]])).sum(axis=0)

    return(DA)

##VOLUMES AND DERIVATIVES

def Compute_Volume_manifold(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = np.cross(Coords[:,1],Coords[:,2],axis=1)
    dots = np.sum(cross_prods*Coords[:,0],axis=1)
    Vol = -np.sum(dots)/6
    return(Vol)

def Compute_Volume_manifold_torch(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1],Coords[:,2],axis=1)
    dots = torch.sum(cross_prods*Coords[:,0],axis=1)
    Vol = -torch.sum(dots)/6
    return(Vol)

def Compute_Volume_manifold_sequential(Verts,Faces):
    Volume = np.zeros(len(Faces))
    for i,face in enumerate(Faces) :
        index = Faces[i,[0,1,2]]
        Coords = Verts[index]
        inc = np.linalg.det(Coords)
        Volume [i]-=inc
    Volume/=6
    return(np.sum(Volume))

def compute_volume_cells(Mesh):
    volumes = {m:0 for m in Mesh.materials}
    for i,face in enumerate(Mesh.faces) :
        index = Mesh.f[i,[0,1,2]]
        Coords = Mesh.v[index]
        inc = np.linalg.det(Coords)
        volumes[face.material_1]+=inc
        volumes[face.material_2]-=inc
        
    for key in volumes : 
        volumes[key]=volumes[key]/6
    return(volumes)

def compute_volume_derivative_dict(Mesh):
    Verts, Faces, Faces_label = Mesh.v, Mesh.f[:,:3], Mesh.f[:,3:]
    materials = Mesh.materials
    
    DV = {key:np.zeros((len(Verts),3)) for key in materials}

    Coords = Verts[Faces]
    X,Y,Z = Coords[:,0],Coords[:,1],Coords[:,2]
    Cross_XY = np.cross(X,Y)/6
    Cross_YZ = np.cross(Y,Z)/6
    Cross_ZX = np.cross(Z,X)/6
    Faces_material = {mat:np.zeros(len(Faces)) for mat in materials}
    for n in materials:
        Faces_material[n][Faces_label[:,0]==1]=1
        Faces_material[n][Faces_label[:,1]==1]=-1

    List_indices_faces_per_vertices_pos_x={key:[[] for i in range(len(Verts))] for key in materials}
    List_indices_faces_per_vertices_pos_y={key:[[] for i in range(len(Verts))] for key in materials}
    List_indices_faces_per_vertices_pos_z={key:[[] for i in range(len(Verts))] for key in materials}
    List_indices_faces_per_vertices_neg_x={key:[[] for i in range(len(Verts))] for key in materials}
    List_indices_faces_per_vertices_neg_y={key:[[] for i in range(len(Verts))] for key in materials}
    List_indices_faces_per_vertices_neg_z={key:[[] for i in range(len(Verts))] for key in materials}
    
    for i, face in enumerate(Faces):
        i_x,i_y,i_z = Faces[i]
        a,b = Faces_label[i]
        #print(i_x,i_y,i_z)
        List_indices_faces_per_vertices_pos_x[a][i_x].append(i)
        List_indices_faces_per_vertices_pos_y[a][i_y].append(i)
        List_indices_faces_per_vertices_pos_z[a][i_z].append(i)
        
        List_indices_faces_per_vertices_neg_x[b][i_x].append(i)
        List_indices_faces_per_vertices_neg_y[b][i_y].append(i)
        List_indices_faces_per_vertices_neg_z[b][i_z].append(i)
    
    for n in tqdm(materials):
        for iv in range(len(Verts)):
            DV[n][iv] = np.vstack((Cross_YZ[List_indices_faces_per_vertices_pos_x[n][iv]],
                                      Cross_ZX[List_indices_faces_per_vertices_pos_y[n][iv]],
                                      Cross_XY[List_indices_faces_per_vertices_pos_z[n][iv]],
                                      - Cross_YZ[List_indices_faces_per_vertices_neg_x[n][iv]],
                                      - Cross_ZX[List_indices_faces_per_vertices_neg_y[n][iv]],
                                      - Cross_XY[List_indices_faces_per_vertices_neg_z[n][iv]])).sum(axis=0)

    return(DV)


##ANGLES

def compute_angles_tri(Mesh,unique=True):
    ##We compute the angles at each trijunctions. If we fall onto a quadrijunction, we skip it
    
    dict_length={}
    dict_angles={}
    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 : 
            face = edge.incident_face
            Faces = [face]
            sources=[edge.origin.key-edge.destination.key]
            Normals = [face.normal]
            materials = [[face.material_1,face.material_2]]
            
            for neighbor in edge.twin : 
                face_attached = Mesh.half_edges[neighbor].incident_face
                Faces.append(face_attached)
                sources.append(Mesh.half_edges[neighbor].origin.key-Mesh.half_edges[neighbor].destination.key)
                materials.append([face_attached.material_1,face_attached.material_2])
                Normals.append(face_attached.normal)
        

            regions_id = np.array(materials)
            if len(regions_id)!=3 : 
                continue
                ## If we fall onto a quadrijunction, we skip it. 

            normals = np.array(Normals).copy()
            
            if regions_id[0,0]==regions_id[1,0]:
                regions_id[1]=regions_id[1][[1,0]]
                normals[1]*=-1
            elif regions_id[0,1]==regions_id[1,1]:
                regions_id[1]=regions_id[1][[1,0]]
                normals[1]*=-1
                
            if regions_id[0,0]==regions_id[2,0]:
                regions_id[2]=regions_id[2][[1,0]]
                normals[2]*=-1
            elif regions_id[0,1]==regions_id[2,1]:
                regions_id[2]=regions_id[2][[1,0]]
                normals[2]*=-1

            pairs = [[0,1],[1,2],[2,0]]

            for i,pair in enumerate(pairs) : 
                i1,i2 = pair
                #if np.isnan(np.arccos(np.dot(normals[i1],normals[i2]))) : 
                #    print("Isnan")
                #if np.dot(normals[i1],normals[i2])>1 or np.dot(normals[i1],normals[i2])<-1 : 
                #    print("Alert",np.dot(normals[i1],normals[i2]))
                angle = np.arccos(np.clip(np.dot(normals[i1],normals[i2]),-1,1))

                if regions_id[i1][1]==regions_id[i2][0]:
                    e,f,g=regions_id[i1][0],regions_id[i1][1],regions_id[i2][1]
                    
                elif regions_id[i1][0]==regions_id[i2][1]:
                    e,f,g=regions_id[i2][0],regions_id[i2][1],regions_id[i1][1]

                dict_angles[(min(e,g),f,max(e,g))]=dict_angles.get((min(e,g),f,max(e,g)),[])
                dict_angles[(min(e,g),f,max(e,g))].append(angle)
                dict_length[(min(e,g),f,max(e,g))]=dict_length.get((min(e,g),f,max(e,g)),0)
                dict_length[(min(e,g),f,max(e,g))]+=(edge.length)
                if not unique : 
                    dict_angles[(min(e,g),f,max(e,g))]=dict_angles.get((min(e,g),f,max(e,g)),[])
                    dict_angles[(min(e,g),f,max(e,g))].append(angle)
                    dict_length[(min(e,g),f,max(e,g))]=dict_length.get((min(e,g),f,max(e,g)),0)
                    dict_length[(min(e,g),f,max(e,g))]+=(edge.length)

    dict_mean_angles = {}
    dict_mean_angles_deg = {}
    for key in dict_angles.keys(): 
        dict_mean_angles[key]=np.mean(dict_angles[key])
        dict_mean_angles_deg[key]=np.mean(dict_mean_angles[key]*180/np.pi)
        
    return(dict_mean_angles,dict_mean_angles_deg,dict_length)


