"""Module dedicated to the representation of a DCEL graph."""
# Define the structures of a double connected edge list (https://en.wikipedia.org/wiki/Doubly_connected_edge_list)
from dataclasses import dataclass, field
import math
import pickle
from subprocess import list2cmdline
import numpy as np 
from fractions import Fraction
import struct
import warnings

from dw2d.io import load_rec, save_rec
from dw2d._matplotlib import pyplot as plt  # imported on first use: matplotlib is in the `viewing` extra
@dataclass
class Vertex:
    """Vertex in 2D"""

    x: float = 0.0
    y: float = 0.0



@dataclass
class HalfEdge:
    """Half-Edge of a DCEL graph"""

    origin: Vertex = None
    destination: Vertex = None
    twin: "HalfEdge" = None
    incident_face: "Face" = None
    prev: "HalfEdge" = None
    next: "HalfEdge" = None
    attached: dict = field(default_factory=dict)

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

    def compute_angle_with_next(self): 
        v1 = self.return_vector()
        v2 = self.next.return_vector()
        product = np.dot(v1,v2)
        angle = np.arccos(product)
        return(angle)

    def compute_angle_with_prev(self): 
        v1 = self.return_vector()
        v2 = self.prev.return_vector()
        product = np.dot(v1,-v2)
        angle = np.arccos(product)
        return(angle)

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
    _closed: bool = False

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

    def find_triple_point_edge(self):
        he = self.first_half_edge()
        b=he
        #return b such that b.origin is a triple point
        #return an error is there is none
        while True :
            previous_incident_face = b.twin.incident_face.attached['key']
            b=b.next
            next_incident_face = b.twin.incident_face.attached['key']
            if previous_incident_face != next_incident_face :
                return(b,True)
            if b.attached['key']==he.attached['key']:
                return(None,False)

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
                return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

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

def Clean_mesh(Seg):
    """The mesh ``(Verts, Edges)``: vertices ``(V, 2)`` in (row, col) pixels, edges ``[v1, v2, m1, m2]``.

    Orientation (since 0.8.0, the rule of dw3d's triangles): ``m1 < m2`` on every edge, and each edge is
    the positively oriented boundary of its first material ``m1`` in the index frame ``(row, col)``:
    ``m1`` lies on the **left** of ``v1 -> v2`` in ``(row, col)`` (on its right in ``(x, y) = (col, row)``).
    Each material's edges, taken as stored where it is ``m1`` and reversed where it is ``m2``, run
    counter-clockwise around it in ``(row, col)``. On an image-border edge ``m1`` is the medium (0), outside
    the image. Until 0.8.0 every edge was stored the other way round (``m1`` on the right in ``(row, col)``);
    the vertices, the edge set and the materials are unchanged. ``tests/test_edge_orientation_contract.py``
    checks it on every edge of the golden cases and the shipped masks, with and without relocation and
    projection.
    """
    Verts, Edges, Nodes_linked = retrieve_mesh_multimaterial_multitracker_format(Seg.Delaunay_Graph,Seg.Map_end)
    if len(Edges) == 0:
        # A mesh without an edge (every cell too small to hold a triangle, or a mask without a cell): returned
        # empty, with its column counts, since 0.8.0; before, ``reorient_edges`` raised IndexError.
        return(np.asarray(Verts, dtype=np.float64).reshape(-1, 2), np.zeros((0, 4), dtype=np.int64))


    for i, f in enumerate(Edges):
        if f[2]>f[3]:
            Edges[i]=Edges[i,[1,0,3,2]]
            Nodes_linked[i]=Nodes_linked[i][[1,0]]

    # ``reorient_edges`` puts ``m1`` on the left of ``v1 -> v2`` in (row, col). Until 0.8.0 every edge was then
    # reversed here, which made the 2-D rule the opposite of the 3-D one in index order.
    Edges = reorient_edges(Edges,Seg,Nodes_linked)

    return(Verts,Edges)





class DCEL_Data:
    """DCEL Graph containing faces, half-edges and vertices."""

    def __init__(self,Verts,Edges,allow_several_loops=False,image_border_vertices=None):
        """Half-edge structure of a 2D multimaterial mesh.

        ``allow_several_loops`` (default False): see ``create_halfedges_from_edges``; when True, a
        material may have several boundary loops (one ``Face`` per loop, same key).

        ``image_border_vertices`` (default None): the indices of the vertices on the image border, kept as
        ``self.image_border_vertices`` (a sorted int64 array, or None when not given). ``return_dcel`` passes
        the mark ``geometry_reconstruction_2d`` takes from the triangulation's structure
        (``image_border_vertices``), before any vertex moves; foambryo's 2D inference reads it to exclude
        the junctions on the border.
        """
        
        self.v = Verts
        self.e = Edges
        self.image_border_vertices = None if image_border_vertices is None else \
            np.unique(np.asarray(image_border_vertices, dtype=np.int64))
        
        Vertices_list = make_vertices_list(Verts)

        Halfedges_list,Faces_list=create_halfedges_from_edges(Edges,Vertices_list,allow_several_loops)

        self.vertices = Vertices_list
        self.faces = Faces_list
        self.half_edges = Halfedges_list

        
    def plot_polyline(self,figsize=(7,7)): 
        plot_DCEL_polyline(self.v,self.e,figsize=figsize)
        
    def plot_bw(self,figsize=(7,7)):
        plot_DCEL_bw(self.v,self.e,figsize=figsize)
        
    def compute_angles(self):
        return(compute_angles(self.faces))

    def compute_area_faces(self): 
        return(compute_area_faces(self.faces))

    def plot_local_curvature(self,labels):
        return(plot_local_curvature(self.faces,labels))

    def find_trijunctions(self): 
        angles_dict = self.compute_angles()
        return(find_trijunctions(angles_dict))
    
    def compute_curvatures(self):
        return(compute_curvatures(self.faces))

    def find_interfaces(self): 
        curvatures_dict = self.compute_curvatures()
        return(find_interfaces(curvatures_dict))

    def compute_lengths(self): 
        return(compute_lengths(self.faces))

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
LOAD 
THE 
SEGMENTATION 
WITH 
A 
HALFEDGE
DATA
STRUCTURE
"""



def make_vertices_list(Verts): 
    Vertices_list = []
    for vertex_coords in Verts : 
        x,y = vertex_coords
        Vertices_list.append(Vertex(x=x,y=y))
    return(Vertices_list)

        
def separate_edges(Edges): 
    n_towers = np.amax(Edges[:,[2,3]])+1
   
    Occupancy=np.zeros(n_towers)
    Dict={}
    
    for edge in Edges : 
        _,_,num1,num2 = edge
        if num1!=-1:
            if Occupancy[num1]==0:
                Dict[num1]=[edge[[0,1]]]
                Occupancy[num1]+=1
            else : 
                Dict[num1].append(edge[[0,1]])
            
        if num2!=-1:
            if Occupancy[num2]==0:
                Dict[num2]=[edge[[0,1]]]
                Occupancy[num2]+=1
            else : 
                Dict[num2].append(edge[[0,1]])
            
    Edges_separated={}
    for key in Dict.keys():
        Edges_separated[key]=(np.array(Dict[key]))
        
    return(Edges_separated)




def separate_edges_with_edge_idx(Edges): 
    n_towers = np.amax(Edges[:,[2,3]])+1
   
    Occupancy=np.zeros(n_towers)
    Dict={}
    
    for idx,edge in enumerate(Edges) : 
        _,_,num1,num2 = edge
        if num1!=-1:
            if Occupancy[num1]==0:
                Dict[num1]=[[edge[0],edge[1],idx]]
                Occupancy[num1]+=1
            else : 
                Dict[num1].append([edge[0],edge[1],idx])
            
        if num2!=-1:
            if Occupancy[num2]==0:
                Dict[num2]=[[edge[0],edge[1],idx]]
                Occupancy[num2]+=1
            else : 
                Dict[num2].append([edge[0],edge[1],idx])
            
    Edges_separated={}
    for key in sorted(Dict.keys()):
        Edges_separated[key]=(np.array(Dict[key]))
        
    return(Edges_separated)





# Use Green's theorem to compute the area
# enclosed by the given contour.
def Green_area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    x1,y1 = vs[0]
    dx = x1-x0
    dy = y1-y0
    a += 0.5*(y0*dx - x0*dy)
    
    return a

def reverse_closed_line(list_halfedges):
    """The closed line ``list_halfedges`` (rows ``[v1, v2, edge_idx]``, each ending where the next begins) run the
    other way: the same cycle of origins, read backwards, each row going to the next row's origin.

    Row ``i`` of the result goes from origin ``o_i`` to ``o_{i+1}``, where ``o`` is the reversed sequence of
    origins, rolled so that the result's origins are those of ``list_halfedges[::-1]``. Until 0.8.0 a loop was
    reversed as ``list_halfedges[::-1]`` alone, which kept each row's direction: every half-edge of a reversed
    loop then ended where its ``next`` did not begin, and the walks along a face (``compute_lengths``,
    ``compute_curvatures``) cut its interfaces one edge off their junctions. The origins, hence the areas and
    the angles at the junctions, are those of the former reversal.
    """
    backwards = np.asarray(list_halfedges)[::-1]
    return np.roll(backwards, -1, axis=0)[:, [1, 0, 2]]


def make_closed_oriented_closed_line(l,verts,is_zero=False): 
    
    ###ICI : Problème dans la manière dont je ferme la ligne : 
    ### 1 -> à un moment les vertices ne sont plus dans l'ordre, donc les halfedges se mettent a faire des trucs bizarres
    ### 2 -> On mesure l'aire n'importe comment : On a besoin des vertices pour cela


    idx_keys = {nv:[] for nv in np.unique(l)}
    #for each index (vert), we will put the elements of l containing this index on a list : 
    for i in range(len(l)): 
        a,b,_ = l[i]
        idx_keys[a].append(i)
        idx_keys[b].append(i)

    closed_line = []
    visited_lines = {i:0 for i in range(len(l))}

    closed_line.append(0)
    visited_lines[0]=1

    start_idx = l[0,0]
    end_idx = l[closed_line[-1],1]

    for i in range(len(l)-1): 
        a,b = idx_keys[end_idx]
        if visited_lines[a]==0 : 
            closed_line.append(a)
            visited_lines[a]=1
            if end_idx == l[a,0] :
                end_idx = l[a,1]
            else : 
                end_idx = l[a,0]
        elif visited_lines[b]==0 : 
            closed_line.append(b)
            visited_lines[b]=1
            if end_idx == l[b,0] :
                end_idx = l[b,1]
            else : 
                end_idx = l[b,0]
        else : 
            print("error")
            break
    assert end_idx ==start_idx
    list_halfedges = l[closed_line]
    va,vb,_ = list_halfedges[0]
    for i in range(1,len(list_halfedges)):
        vc,vd,e = list_halfedges[i]
        if vb == vc : 
            va,vb,_ = list_halfedges[i]
            continue
        else : 
            list_halfedges[i] = list_halfedges[i,[1,0,2]]
        va,vb,_ = list_halfedges[i]

    indices = list_halfedges[:,0]
    area = Green_area(verts[indices])
    #print(list_halfedges)
    #print(closed_line)
    #area = Green_area(l[closed_line,:2])
    
    if is_zero : 
        if area >0 : 
            list_halfedges = reverse_closed_line(list_halfedges)
    else : 
        if area <0 : 
            list_halfedges = reverse_closed_line(list_halfedges)

    indices = list_halfedges[:,0]
    area = Green_area(verts[indices])
    #print("Area:", area)
    #assert Green_area(verts[l[closed_line,:2])>0
    return(list_halfedges)#closed_line)

def _split_into_loops(l):
    """Split one material's edges ``l`` (rows ``[v1, v2, edge_idx]``) into vertex-connected loops.

    Used only by the ``allow_several_loops`` path of ``create_halfedges_from_edges``. Returns the
    sub-arrays of ``l`` in order of first appearance, so a single loop returns ``[l]`` unchanged.
    """
    adj = {}
    for i, (a, b, _e) in enumerate(l):
        adj.setdefault(int(a), []).append(i)
        adj.setdefault(int(b), []).append(i)
    seen_rows = np.zeros(len(l), dtype=bool)
    loops = []
    for i in range(len(l)):
        if seen_rows[i]:
            continue
        stack, rows = [i], []
        seen_rows[i] = True
        while stack:
            r = stack.pop()
            rows.append(r)
            for v in (int(l[r, 0]), int(l[r, 1])):
                for r2 in adj[v]:
                    if not seen_rows[r2]:
                        seen_rows[r2] = True
                        stack.append(r2)
        loops.append(l[sorted(rows)])
    return loops


def _orient_several_loops(loops, verts, is_zero):
    """Orient each loop of one material so the material lies on a consistent side.

    Each loop is first oriented by the legacy rule (``make_closed_oriented_closed_line``: positive
    Green area for a cell, negative for the medium 0, i.e. every loop treated as that material's
    outer boundary). A loop nested inside an odd number of the material's other loops is then
    reversed: for a cell that is a hole (the cell lies outside it); for the medium it is a lumen
    (the medium lies inside it). Loops of one material are vertex-disjoint once its pinch vertices
    are repaired, so nesting is decided by testing one vertex of a loop against the others.
    """
    oriented = [make_closed_oriented_closed_line(lp, verts, is_zero) for lp in loops]
    polygons = [verts[o[:, 0]] for o in oriented]
    out = []
    for i, o in enumerate(oriented):
        probe = verts[int(o[0, 0])]
        depth = sum(1 for j, poly in enumerate(polygons) if j != i and point_strictly_inside_polygon(probe, poly))
        out.append(reverse_closed_line(o) if depth % 2 == 1 else o)
    return out


def point_strictly_inside_polygon(point: np.ndarray | tuple[float, float], polygon: np.ndarray) -> bool:
    """Whether ``point`` lies inside the polygon ``polygon``, by the even-odd rule, decided exactly.

    ``point`` is ``(x, y)``; ``polygon`` is an ``(n, 2)`` array of its vertices in order, the edge from
    the last vertex back to the first included (a repeated closing vertex adds an edge of length zero,
    which changes nothing). Counts the edges that a horizontal ray from ``point`` towards ``+x`` crosses:
    an edge counts when exactly one of its ends lies strictly above ``y`` (half-open, so a ray through a
    vertex counts it once), and the side of the crossing is the sign of a 2x2 determinant computed with
    ``Fraction`` from the float coordinates, so rounding never decides it.

    Defined for a point off the polygon's boundary, which is how ``_orient_several_loops`` uses it: the
    probe is a vertex of another loop of the same material, and those loops share no vertex and the
    mesh's edges do not touch. For such a point it gives the answer of ``matplotlib.path.Path.contains_point``,
    which this replaced (since 0.8.0) so that the reconstruction does not need matplotlib, wherever the
    latter's floating-point test is right; ``tests/two_dimensional/test_point_in_polygon.py`` compares the
    two on every loop of the golden cases and the shipped masks.
    """
    px, py = float(point[0]), float(point[1])
    xs, ys = np.asarray(polygon, dtype=np.float64)[:, 0], np.asarray(polygon, dtype=np.float64)[:, 1]
    xs_next, ys_next = np.roll(xs, -1), np.roll(ys, -1)
    straddling = (ys > py) != (ys_next > py)
    x, y = Fraction(px), Fraction(py)
    inside = False
    for x1, y1, x2, y2 in zip(xs[straddling], ys[straddling], xs_next[straddling], ys_next[straddling], strict=True):
        x1, y1, x2, y2 = Fraction(float(x1)), Fraction(float(y1)), Fraction(float(x2)), Fraction(float(y2))
        # The crossing's abscissa x1 + (x2 - x1) (y - y1) / (y2 - y1) exceeds x exactly when this determinant
        # has the sign of y2 - y1 (nonzero here, since the edge straddles y).
        determinant = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
        if determinant != 0 and (determinant > 0) == (y2 > y1):
            inside = not inside
    return inside


def create_halfedges_from_edges(Edges,Vertices_list,allow_several_loops=False): 
    """Build the half-edges and faces of the 2D multimaterial mesh.

    Legacy contract (``allow_several_loops=False``, the default and the published behaviour): every
    material's edges must form exactly one simple closed loop, asserted. With
    ``allow_several_loops=True`` (used after the pinch-vertex repair, on by default in
    ``geometry_reconstruction_2d``) a material whose edges form several vertex-disjoint loops gets one ``Face`` per loop,
    all carrying the material's key, oriented by ``_orient_several_loops``; a material with one loop
    goes through exactly the legacy statements, so single-loop meshes are built identically.
    Failure mode: a vertex of degree other than 2 in a material's edges (an unrepaired pinch or a
    dangling edge) still fails the assertion.

    Each loop is oriented by its area (``make_closed_oriented_closed_line``), whatever the stored direction
    of the edges, but the half-edge it starts from depends on that direction. The loops are traced from each
    edge's second vertex, the direction dw2d stored until 0.8.0, so that a mesh in the orientation of 0.8.0
    gives the half-edges, hence the lengths and curvatures, of the same mesh before 0.8.0 bit for bit.
    """

    Edge_to_halfedges = {i:[] for i in range(len(Edges))}
    polylines = separate_edges_with_edge_idx(np.asarray(Edges)[:, [1, 0, 2, 3]])
    halfedge_idx = 0
    HalfEdges_list=[]
    Faces_list = []
    verts = np.array([[v.x,v.y ]for v in Vertices_list])

    for cell_idx in polylines:
        #print("Cell_idx",cell_idx)
        _,c=np.unique(polylines[cell_idx][:,:2],return_counts=True)
        assert np.unique(c)==2
        l = polylines[cell_idx]
        loops = _split_into_loops(l) if allow_several_loops else [l]
        if len(loops) == 1:
            closed_lines = [make_closed_oriented_closed_line(l,verts,cell_idx==0)]
        else:
            closed_lines = _orient_several_loops(loops, verts, cell_idx==0)
        for closed_line in closed_lines:
            #print(closed_line)
            #
            #print(Green_area(verts[closed_line]))
            starting_idx = halfedge_idx
            for i,n in enumerate(closed_line):
                v1,v2,edge_idx = n#l[n] 
                ##create_halfedge_routine
                HalfEdges_list.append(HalfEdge(attached = {'key':halfedge_idx},origin = Vertices_list[v1],destination = Vertices_list[v2]))    
                Edge_to_halfedges[edge_idx].append(halfedge_idx)
                halfedge_idx+=1

            for i,n in enumerate(closed_line): 
                if i ==0 : 
                    HalfEdges_list[starting_idx+i].prev = HalfEdges_list[halfedge_idx-1]
                    HalfEdges_list[starting_idx+i].next = HalfEdges_list[starting_idx+i+1]
                elif i==len(closed_line)-1 : 
                    assert halfedge_idx-1==starting_idx+i
                    HalfEdges_list[starting_idx+i].prev = HalfEdges_list[starting_idx+i-1]
                    HalfEdges_list[starting_idx+i].next = HalfEdges_list[starting_idx]
                else : 
                    HalfEdges_list[starting_idx+i].next = HalfEdges_list[starting_idx+i+1]
                    HalfEdges_list[starting_idx+i].prev = HalfEdges_list[starting_idx+i-1]

            Faces_list.append(Face(attached = {'key':cell_idx},outer_component=HalfEdges_list[starting_idx]))
            assert Faces_list[-1].closed()
            for i,n in enumerate(closed_line): 
                HalfEdges_list[starting_idx+i].incident_face = Faces_list[-1]

    #Unit test
    for key in Edge_to_halfedges : 
        assert(len(Edge_to_halfedges[key])==2)

    for key in Edge_to_halfedges : 
        a,b = Edge_to_halfedges[key]
        HalfEdges_list[a].twin=HalfEdges_list[b]
        HalfEdges_list[b].twin=HalfEdges_list[a]
    return(HalfEdges_list,Faces_list)



"""
COMPUTE GEOMETRIC QUANTITIES WITH THIS DATA STRUCTURE
"""


def compute_angles(Faces_list): 
    
    dict_angles = {}
    for f in Faces_list : 
        he = f.first_half_edge()
        b = he

        while True :
            #print(b.twin.incident_face)
            previous_incident_face = b.twin.incident_face.attached['key']
            b=b.next
            if b is None : 
                break

            next_incident_face = b.twin.incident_face.attached['key']

            if previous_incident_face != next_incident_face : 
                tuple_one = (previous_incident_face,b.incident_face.attached['key'],next_incident_face)#unique because we always turn in the same sense
                #e,f,g = tuple_one
                #tup = (min(e,g),f,max(e,g))
                dict_angles[tuple_one]=b.compute_angle_with_prev()
                
            if b.attached['key']==he.attached['key']:
                break

    sorted_dict = {}
    for key in sorted(dict_angles.keys()):
        sorted_dict[key] = dict_angles[key] 

    for key in sorted(dict_angles.keys()):
        a,b,c = key
        sorted_dict[(c,b,a)] = dict_angles[key] 

    return(sorted_dict)


def find_closed_faces(Faces_list): 
    Closed_faces=[]
    for f in Faces_list : 
        if f.closed() : 
            Closed_faces.append(f.attached['key'])
    return(Closed_faces)



def compute_curvatures(Faces_list,eps = 1e-10):
    """Mean curvature of every interface: ``{(a, b): curvature}`` (1/px, unsigned) for each pair ``a < b``.

    The length-weighted mean of the Menger curvature at the interface's interior vertices,
    ``sum_i c_i (l_i^- + l_i^+) / sum_i (l_i^- + l_i^+)``, read along the half-edges of material ``a``'s faces from
    junction to junction: a junction vertex is not interior; on a closed loop without junction (a cell enclosed by
    one material, a pocket of one material inside a cell) every vertex is. Equal, to rounding, to the same sum read
    straight from the mesh's edges: ``tests/test_dcel_curvatures.py`` checks it on every golden case and every
    shipped mask. Until 0.8.0 a closed loop without junction left out its first and last vertices, and the faces
    whose loop had been reversed read every interface one edge off its junctions (see ``reverse_closed_line``).
    """
    Points_interface={}
    for f in Faces_list : 
        if not f.closed() : 
            continue

        he,success = f.find_triple_point_edge()
        if not success : 
            he = f.outer_component

        b=he
        points = []
        face_key = he.incident_face.attached['key']
        current_interface = (face_key,he.twin.incident_face.attached['key'])
        while True : 
            previous_incident_face = b.twin.incident_face.attached['key']
            points.append([b.origin.x,b.origin.y])
            b=b.next
            next_incident_face = b.twin.incident_face.attached['key']

            if (previous_incident_face != next_incident_face)  : 
                points.append([b.origin.x,b.origin.y])
                List = Points_interface.get(current_interface,[])
                List.append(np.array(points.copy()))
                Points_interface[current_interface] = List.copy()
                current_interface=(face_key,next_incident_face)
                points=[]

            if (b.attached['key']==he.attached['key']) :
                if not success :
                    # a closed loop without junction: every vertex is interior, so the loop is closed through its
                    # first two vertices and the Menger curvature is read at each vertex once, the first included
                    points.append([he.origin.x,he.origin.y])
                    points.append([he.next.origin.x,he.next.origin.y])
                    List = Points_interface.get(current_interface,[])
                    List.append(np.array(points.copy()))
                    Points_interface[current_interface] = List.copy()
                break

    Curvatures = {}
    for key in sorted(Points_interface.keys()) : 
        if key[0]>key[1]: 
            continue

        Curvatures[key] = 0

        total_l = []

        for m in range(len(Points_interface[key])):

            p = Points_interface[key][m]
            curvature = 0
            if len(p)>2 : 
                for i in range(1,len(p)-1):
                    l1 = np.linalg.norm(p[i-1]-p[i])
                    l2 = np.linalg.norm(p[i]-p[i+1])
                    l = l1 + l2
                    total_l.append(l)
                    c = Menger_Curvature(p[i-1],p[i],p[i+1])
                    curvature+=c*l
            Curvatures[key]+=curvature

        Curvatures[key]/=max(sum(total_l),eps)
    return(Curvatures)



def Menger_Curvature(Point_1,Point_2,Point_3,eps = 1e-10): 
    A = Area(Point_1,Point_2,Point_3)
    d = np.clip(np.linalg.norm(Point_2-Point_1)*np.linalg.norm(Point_3-Point_2)*np.linalg.norm(Point_3-Point_1),eps,np.inf)
    return(4*A/d)

def Compute_local_curvature(line,Graph,step=1,borders_thresh=0): 
    #borders_thresh=2 : we eliminate first 2 and last 2 points
    curvatures = []
    Verts = Graph.Vertices
    for i in range(step+borders_thresh,len(line)-step-borders_thresh):
        menger_curvature = Menger_Curvature(Verts[line[i-step]],Verts[line[i]],Verts[line[i+step]])
        curvatures.append(menger_curvature)
    curvatures = np.array(curvatures)
    return(curvatures,np.mean(curvatures),np.std(curvatures))

def Area(Point_1,Point_2,Point_3):
    #The result match the one of Area_Heron
    return(np.abs(Point_1[0]*(Point_2[1]-Point_3[1]) + Point_2[0]*(Point_3[1]-Point_1[1]) + Point_3[0]*(Point_1[1]-Point_2[1]))/2)


def find_trijunctions(Angles_dict): 
    #Angle dict : output of compute_angles
    list_trijunctions = list(Angles_dict.keys())

    L = np.sort(np.array(list_trijunctions),axis=1)
    key = np.amax(L)+1
    Keys = L[:,0] + L[:,1]*key + L[:,2]*(key**2) 

    _,index = np.unique(Keys,return_index=True)
    triple_points = L[index]

    return(triple_points)

def find_interfaces(Curvatures_dict): 
    list_interfaces = list(Curvatures_dict.keys())
    
    L = np.sort(np.array(list_interfaces),axis=1)
    key = np.amax(L)+1
    Keys = L[:,0] + L[:,1]*key 
    
    _,index = np.unique(Keys,return_index=True)
    Interfaces = L[index]

    return(Interfaces)



def compute_lengths(Faces_list): 
    """Length of every interface: ``{(a, b): length}`` for each pair of materials ``a < b`` that share edges.

    Read along the half-edges of material ``a``'s faces, from junction to junction (a closed loop without junction,
    a cell enclosed by one material, is read in full). Equal to the sum of the lengths of the edges of the pair, to
    rounding: ``tests/test_dcel_lengths.py`` checks it on every golden case and every shipped mask. Until 0.8.0 the
    two disagreed on most cell-cell interfaces (the faces whose loop had been reversed cut their interfaces one edge
    off their junctions, see ``reverse_closed_line``) and on every closed loop without junction (its last edge was
    left out, except on a mesh of a single material).
    """
    Points_interface={}
    for f in Faces_list : 
        if not f.closed() : 
            continue

        he,success = f.find_triple_point_edge()
        if not success : 
            he = f.outer_component

        b=he
        points = []
        face_key = he.incident_face.attached['key']
        current_interface = (face_key,he.twin.incident_face.attached['key'])

        while True : 
            previous_incident_face = b.twin.incident_face.attached['key']
            points.append([b.origin.x,b.origin.y])
            b=b.next
            next_incident_face = b.twin.incident_face.attached['key']

            if (previous_incident_face != next_incident_face)  : 
                points.append([b.origin.x,b.origin.y])
                List = Points_interface.get(current_interface,[])
                List.append(np.array(points.copy()))
                Points_interface[current_interface] = List.copy()
                current_interface=(face_key,next_incident_face)
                points=[]

            if (b.attached['key']==he.attached['key']) : 
                if not success : 
                    points.append([he.origin.x,he.origin.y])  # the loop's last edge, back to its first vertex
                    List = Points_interface.get(current_interface,[])
                    List.append(np.array(points.copy()))
                    Points_interface[current_interface] = List.copy()
                break

    Lengths = {}
    for key in sorted(Points_interface.keys()): 
        if key[0]>key[1]: 
            continue
        total_l = 0
        for p in Points_interface[key]:
            for i in range(1,len(p)):
                total_l+=np.linalg.norm(p[i-1]-p[i])
        Lengths[key]=total_l     

    return(Lengths)


def renormalize_verts(Verts,Edges): 
    #When the Vertices are only a subset of the faces, we remove the useless vertices and give the new faces
    idx_Verts_used = np.unique(Edges)
    Verts_used = Verts[idx_Verts_used]
    idx_mapping = np.arange(len(Verts_used))
    mapping = dict(zip(idx_Verts_used,idx_mapping))
    def func(x): 
        return([mapping[x[0]],mapping[x[1]]])
    New_Edges = np.array(list(map(func,Edges)))
    return(Verts_used,New_Edges)


def change_idx_cells(faces,mapping):
    #mapping : {key_init:key_end} has to be a bijection
    new_faces = faces.copy()
    for key in mapping : 
        new_faces[:,3][faces[:,3]==key]=mapping[key]
        new_faces[:,4][faces[:,4]==key]=mapping[key]
    return(new_faces)








"""
Plot elements from this data structure
"""

def plot_DCEL_bw(Verts,Edges,figsize=(5,5)): 
    
    Edges_list=separate_edges(Edges)
    fig,ax = plt.subplots(figsize=figsize)
    
    ax.set_facecolor('k')
    for key in Edges_list : 
        line = Edges_list[key]
        for elt in line : 
            
            points = Verts[elt]
            ax.plot(points[:,1],points[:,0],'w')
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()


def plot_DCEL_polyline(Verts,Edges,figsize=(5,5)): 
    Edges_list=separate_edges(Edges)
    fig,ax = plt.subplots(figsize=figsize)
    for key in Edges_list : 
        line = Edges_list[key]
        plot_polylines(line,Verts)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

def plot_line(line,Verts,color=np.random.rand(3)):
    for elmts in line : 
        a,b=elmts
        coords = Verts[[a,b]]
        plt.plot(coords[:,1],coords[:,0],'o-',color=color,linewidth=2, markersize=8)

def plot_line_no_marker(line,Verts,color=np.random.rand(3)):
    print(line)
    for elmts in line : 
        a,b,=elmts
        coords = Verts[[a,b]]
        plt.plot(coords[:,1],coords[:,0],'-',color=color,linewidth=2)
        
def plot_lines(Lines,Verts,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line(line, Verts,color=color)

def plot_polylines_no_marker(Lines,Verts,figsize=8,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line_no_marker(line, Verts,color=color)

def plot_polylines(line,Verts,figsize=8,random_seed=0): 
    #np.random.seed(random_seed)
    color=np.random.rand(3)
    plot_line(line, Verts,color=color)
    
def compute_area_faces(Faces_list):
    
    Area_faces = {}
    for f in Faces_list : 
        key = f.attached['key']
        if not f.closed() : 
            continue

        he = f.outer_component
        b=he
        points = []
        while True : 
            points.append([b.origin.x,b.origin.y])
            b=b.next
            if b.attached['key']==he.attached['key']: 
                break
        
        points = np.array(points)
        # A material with several loops (allow_several_loops) has one Face per loop: their signed
        # areas add (holes are oriented the other way). With one loop this is the legacy value.
        Area_faces[key]=Area_faces.get(key,0)+Green_area(points)
        

    return(Area_faces)
    
    

def Green_area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    x1,y1 = vs[0]
    dx = x1-x0
    dy = y1-y0
    a += 0.5*(y0*dx - x0*dy)
    
    return a



def image_border_vertices(Graph):
    """Indices of the vertices on the image border, from the structure of the Delaunay triangulation.

    dw2d triangulates its points together with the four image corners, so the triangulation's hull is the image
    rectangle, and its hull edges are the edges with a single triangle (``Graph.Lone_edges``). The border vertices
    are their endpoints, read from the triangulation, not from coordinates: they stay the border vertices after the
    junction relocation or the projection moves vertices (neither moves them). Indices into ``Graph.Vertices``,
    hence into the mesh's vertices. Returns a sorted int64 array.
    """
    lone = np.asarray(Graph.Lone_edges, dtype=np.int64).reshape(-1, 2)
    return np.unique(lone)


def retrieve_mesh_multimaterial_multitracker_format(Graph,Map):
    ##Must be used without any filtering operation
    reverse_map ={}
    for key in Map : 
        for node_idx in Map[key] :
            reverse_map[node_idx]=key
    Edges=[]
    Edges_idx = []
    Nodes_linked = []
    for idx in range(len(Graph.Edges)) : 
        nodes_linked = Graph.Nodes_linked_by_Edges[idx]
        

        cluster_1 = reverse_map[nodes_linked[0]]#reverse_map.get(nodes_linked[0],-1)
        cluster_2 = reverse_map[nodes_linked[1]]#reverse_map.get(nodes_linked[1],-2)
        #if the two nodes of the edges belong to the same cluster we ignore them
        #otherwise we add them to the mesh
        if cluster_1 != cluster_2 : 
            
            #If one thing has been filtered we add it to the background
            #If both have been filtered the interface is considered to not exist
            #if cluster_1 == -1 and cluster_2 ==-2 :
            #    continue
            #if cluster_2 == -2 : 
            #    cluster_2 = max(Map.keys())+1 
            #if cluster_1 ==-1 : 
            #    cluster_1 = max(Map.keys())+1 
                
            edge = Graph.Edges[idx]
            cells = [cluster_1,cluster_2]
            Edges.append([edge[0],edge[1],cells[0],cells[1]])
            Edges_idx.append(idx)
            Nodes_linked.append(nodes_linked)

    for idx in range(len(Graph.Lone_edges)):
        edge = Graph.Lone_edges[idx]
        node_linked = Graph.Nodes_linked_by_lone_edges[idx]
        cluster_1 = reverse_map[node_linked]
        #We incorporate all these edges because they are border edges
        if cluster_1 !=0:
            cells = [0,cluster_1]
            Edges.append([edge[0],edge[1],cells[0],cells[1]])
            Edges_idx.append(idx)
            # A border edge has one triangle, the cell's (material cells[1]); it is listed twice so that
            # ``reorient_edges`` can recognise it. Until 0.8.0 this line appended the previous interior edge's
            # pair of triangles (a stale loop variable), so border edges were oriented arbitrarily.
            Nodes_linked.append([node_linked, node_linked])

    return(Graph.Vertices, np.array(Edges),np.array(Nodes_linked))
#Verts,Edges = retrieve_mesh_multimaterial_multitracker_format(DW.Delaunay_Graph,DW.Map_end)


def compute_areas(Verts,Edges): 
    """{material: signed area} of the oriented edges ``[v1, v2, m1, m2]`` (Green's formula, (row, col) frame).

    Each edge counts for ``m1`` as stored and for ``m2`` reversed. With dw2d's orientation since 0.8.0 (each edge the
    positively oriented boundary of ``m1``) a cell's value is its area, positive; the medium's is minus the area it
    surrounds. Before 0.8.0, with every edge stored the other way round, the signs were the opposite.
    """
    Areas = {key:0 for key in np.unique(Edges[:,2:])}
    
    for edge in Edges : 
        i1,i2, m1,m2 = edge
        v1,v2 = Verts[i1],Verts[i2]
        dx = v2[0]-v1[0]
        dy = v2[1]-v1[1]
        Areas[m1]+=0.5*(-v2[1]*dx + v2[0]*dy)
        Areas[m2]-=0.5*(-v2[1]*dx + v2[0]*dy)
    
    return Areas


def compute_curvatures_vertices(Faces_list): 
    Points_face={}

    for f in Faces_list : 
        if not f.closed() : 
            continue

        he,success = f.find_triple_point_edge()
        if not success : 
            he = f.outer_component

        b=he
        points = []
        face_key = he.incident_face.attached['key']
        current_interface = (face_key,he.twin.incident_face.attached['key'])
        while True : 
            points.append([b.origin.x,b.origin.y])
            b=b.next

            if (b.attached['key']==he.attached['key']) : 
                break
            
        points.append([he.origin.x,he.origin.y])
        points.append([he.next.origin.x,he.next.origin.y])
        Points_face[face_key]=np.array(points.copy())

    Curvatures_points = {}
    Curvatures_edges = {}
    Curvatures_lines = {}
    for key in sorted(Points_face.keys()) :         

        p = Points_face[key]
        Curvatures_points[key]=np.zeros(len(p))
        Curvatures_edges[key]=np.zeros(len(p)-1)
        Curvatures_lines[key]=[]
        if len(p)>2 : 

            total_l = []

            for i in range(1,len(p)-1):
                l1 = np.linalg.norm(p[i-1]-p[i])
                l2 = np.linalg.norm(p[i]-p[i+1])
                l = l1 + l2
                total_l.append(l)
                c = Menger_Curvature(p[i-1],p[i],p[i+1])
                Curvatures_points[key][i] = c
            
            Curvatures_points[key][0] = Curvatures_points[key][1]
            Curvatures_points[key][-1] = Curvatures_points[key][-2]

            for i in range(len(p)-1):
                Curvatures_edges[key][i] = ( Curvatures_points[key][i] + Curvatures_points[key][i+1] ) / 2
            for i in range(len(p)-1):
                Curvatures_lines[key].append(trace_line(p[i][0], p[i][1], p[i+1][0], p[i+1][1]))
        else : 
            Curvatures_edges[key]=[0]
            Curvatures_lines[key]=[trace_line(p[0][0], p[0][1], p[1][0], p[1][1])]
    return(Curvatures_points,Curvatures_edges,Curvatures_lines)

def plot_local_curvature(Faces_list,labels): 

    Curvatures_points,Curvatures_edges,Curvatures_lines = compute_curvatures_vertices(Faces_list)

    All_l = []
    for l in list(Curvatures_edges.values()) : 
        for sl in l : 
            All_l.append(sl)
    vals = np.hstack(All_l)

    alpha = 0.05
    mins,maxs = np.quantile(vals,alpha),np.quantile(vals,1-alpha)
    for key in Curvatures_edges :
        for m in range(len(Curvatures_edges[key])) :
            Curvatures_edges[key] = np.clip(Curvatures_edges[key],mins,maxs)

    A = np.zeros(labels.shape)
    for key in Curvatures_lines : 
        for i in range(len(Curvatures_lines[key])):
            for m in range(len(Curvatures_lines[key][i])):
                for p in Curvatures_lines[key][i] :
                    # an end on the image's outer pixel corner rounds one pixel outside: drawn on the border pixel
                    A[min(max(p[0], 0), A.shape[0] - 1), min(max(p[1], 0), A.shape[1] - 1)] = Curvatures_edges[key][i]

    plt.figure(figsize=(7,7))
    plt.imshow(A)
    plt.colorbar()

def trace_line(x0, y0, x1, y1):
    """The pixels of the segment `(x0, y0) -> (x1, y1)` (row, column), both ends included: Bresenham's line.

    `dw2d.Mask_reconstruction.trace_line`, which rounds each end to the nearest pixel centre first. Before 0.8.0 this
    module had its own copy, which walked by unit steps until it met the far end exactly, so it never ended (and grew
    its list of points without bound) when the two ends differed by a non-integer, as they do after the default
    junction relocation: `plot_local_curvature` hung on every relocated mesh. On integer ends the result is unchanged.
    """
    from dw2d.Mask_reconstruction import trace_line as _trace_line_on_the_lattice

    return _trace_line_on_the_lattice((x0, y0), (x1, y1))











# The `.rec` reader and writer of dw2d 0.1.5 and earlier, kept through 0.8.x as deprecated aliases of `dw2d.io` (the
# format's own names, as in `dw3d.io`) and removed in 0.9. They return what they returned before: the edges as one
# (m, 4) int64 array and, from the readers, the counts `[n_vertices, n_edges]`.
def _deprecated(old_name, new_name):
    warnings.warn(
        f"dw2d.Dcel.{old_name} is deprecated and will be removed in 0.9; use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _load_verts_and_edges(filename):
    """`(verts, edges)` with `edges` `(m, 4)` int64, as the 0.1.5 readers returned them, from `dw2d.io.load_rec`."""
    points, segments, labels = load_rec(filename)
    return points, np.hstack((segments, labels)).astype(np.int64, copy=False)


def write_mesh_bin(filename, Verts, Edges):
    """Deprecated: `dw2d.io.save_rec(filename, Verts, Edges, binary_mode=True)`."""
    _deprecated("write_mesh_bin", "dw2d.io.save_rec(filename, points, segments, labels, binary_mode=True)")
    Edges = np.asarray(Edges)
    save_rec(filename, Verts, Edges[:, :2], Edges[:, 2:], binary_mode=True)


def write_mesh_text(filename, Verts, Edges):
    """Deprecated: `dw2d.io.save_rec(filename, Verts, Edges)` (text). Coordinates are now written exactly, not to 5 decimals."""
    _deprecated("write_mesh_text", "dw2d.io.save_rec(filename, points, segments, labels)")
    Edges = np.asarray(Edges)
    save_rec(filename, Verts, Edges[:, :2], Edges[:, 2:], binary_mode=False)


def open_mesh_multitracker(type_file, filename):
    """Deprecated: `dw2d.io.load_rec(filename)`, which detects binary or text itself.

    Returns `(verts, edges, [n_vertices, n_edges])` as before; `type_file` is no longer read, and `filename` has no
    default any more (it named a path on one author's machine).
    """
    _deprecated("open_mesh_multitracker", "dw2d.io.load_rec(filename)")
    verts, edges = _load_verts_and_edges(filename)
    return(verts, edges, np.array([len(verts), len(edges)]))


def read_rec_file_bin(filename):
    """Deprecated: `dw2d.io.load_rec(filename)`. Returns `(verts, edges, [n_vertices, n_edges])` as before."""
    _deprecated("read_rec_file_bin", "dw2d.io.load_rec(filename)")
    verts, edges = _load_verts_and_edges(filename)
    return(verts, edges, np.array([len(verts), len(edges)]))


def read_rec_file_num(filename, offset=0):
    """Deprecated: `dw2d.io.load_rec(filename)`. Returns `(verts, edges, [n_vertices, n_edges])` as before, the
    materials shifted by `offset`."""
    _deprecated("read_rec_file_num", "dw2d.io.load_rec(filename)")
    verts, edges = _load_verts_and_edges(filename)
    edges[:, [2, 3]] += offset
    return(verts, edges, np.array([len(verts), len(edges)]))


def compute_normal_Edges(Verts,Edges):
    Pos = Verts[Edges[:,[0,1]]]
    Sides_1 = Pos[:,1]-Pos[:,0]
    Sides_1_z = np.zeros((len(Sides_1),3))
    Sides_1_z[:,:2]=Sides_1
    
    Z = np.zeros(Sides_1_z.shape)
    Z[:,2]=1
    
    Normal_edges = np.cross(Z,Sides_1_z,axis=1)[:,:2]
    Norms = np.linalg.norm(Normal_edges,axis=1)#*(1+1e-8)
    Normal_edges/=(np.array([Norms]*2).transpose())
    return(Normal_edges)


def reorient_edges(Edges,Seg,Nodes_linked):
       
    #Thumb rule for all the Edges
    
    Normals = compute_normal_Edges(Seg.Delaunay_Graph.Vertices,Edges)
    
    P = Seg.Delaunay_Graph.Vertices[Edges[:,:2]]
    Centroids_Edges = np.mean(P,axis=1)
    Centroids_nodes = np.mean(Seg.Delaunay_Graph.Vertices[Seg.Delaunay_Graph.Tris[Nodes_linked[:,0]]],axis=1)
    Vectors = Centroids_nodes-Centroids_Edges
    Norms = np.linalg.norm(Vectors,axis=1)
    Vectors[:,0]/=Norms
    Vectors[:,1]/=Norms

    #print(Vectors)
    #print(Normals)
    Dot_product = np.sum(np.multiply(Vectors,Normals),axis=1)
    Normals_sign = np.sign(Dot_product)
    # On a border edge the only triangle is the cell's, material m2, not m1's: it must end on the other side.
    Nodes_linked = np.asarray(Nodes_linked)
    Normals_sign[Nodes_linked[:, 0] == Nodes_linked[:, 1]] *= -1
    
    #Reorientation according to the normal sign
    reoriented_Edges = Edges.copy()
    for i,s in enumerate(Normals_sign) : 
        #print(s)
        if s <0 : 
            reoriented_Edges[i]=reoriented_Edges[i][[1,0,2,3]]
            
    return(reoriented_Edges)