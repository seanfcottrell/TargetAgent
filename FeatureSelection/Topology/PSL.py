import numpy as np
import gudhi

def coboundary_nonconstant_0(vertices, edges, charges, F):
    """
    Computes the coboundary matrix d_0 for the class of nonconstant sheaves defined by the LogFC dysregulation of a DEG
    Parameters
    ----------
    vertices: list
    edges: dictionary
    charges: list of gene LogFC 
    F: a dictionary. A key is a simplex S and the value of it is F(S).
    Outputs
    -------
    d_0: d_0
    """   
    # d_0 is the coboundary matrix
    d_0 = np.zeros((len(edges), len(vertices)))
    for edge, idx_and_t in edges.items():
        # v_0 \leq [v_0, v_1], q_1/F([v_0, v_1])
        d_0[idx_and_t[0], edge[0]] = -charges[edge[1]] / F[edge]
        d_0[idx_and_t[0], edge[1]] = charges[edge[0]] / F[edge]
    return d_0

def coboundary_nonconstant_1(edges, faces, charges, F):
    """
    edges: dictionary. keys are edges (tuple) and values are tuples of indices and filtration values. 
    faces: dictionary. keys are faces (tuple) and values are tuples of indices and filtration values.
    charges: a list (or a 1d numpy array) of charges
    F: a dictionary. A key is a simplex S and the value of it is F(S).
    """
    d_1 = np.zeros((len(faces), len(edges)))
    for face, idx_and_t in faces.items():
        face_face0 = (face[1], face[2])
        face_face1 = (face[0], face[2])
        face_face2 = (face[0], face[1])
        # restriction of [v_0, v_1] \leq [v_0, v_1, v_2], given by F([v_0, v_1])q_2/F([v_0, v_1, v_2])
        d_1[idx_and_t[0], edges[face_face0][0]] = charges[face[0]]*F[face_face0]/F[face]
        d_1[idx_and_t[0], edges[face_face1][0]] = -charges[face[1]]*F[face_face1]/F[face]
        d_1[idx_and_t[0], edges[face_face2][0]] = charges[face[2]]*F[face_face2]/F[face]
    return d_1

class PSL():
    def __init__(self, adj, charges = None, radii = None, p = 0):
        """
        adj: graph structure indicating connections in gene ppi network
        charges: a 1d np array of gene specific LogFC labels (-1,1) to emphasize each gene's dysregulation for PPI topological perturbations  
        radii: filtration radii for inverse rips filtration (in STRING larger weight -> stronger relation)
        """
        self.p = p
        self.simplex_tree = None
        self.radii = radii
        self.F = {} # Will be the dictionary that stores values of F
        self.adj = adj
        if np.any(charges != None):
            self.charges = charges

    def build_filtration(self):
        """
        build 2d clique complex from gene PPI network
        """
        self.simplex_tree = gudhi.SimplexTree()

        for node in self.adj.nodes():
            self.simplex_tree.insert([node], filtration=0.0)
        for u, v, data in self.adj.edges(data=True):
            filtration_value = self.adj[u][v]['weight']  
            self.simplex_tree.insert([u, v], filtration=filtration_value)

        self.simplex_tree.expansion(2) # 2d clique complex for l0, l1 PSL

        # Generate values of F from the network distances (if disconnected, dist = inf)
        for simplex, _ in self.simplex_tree.get_filtration():
            if len(simplex) == 1:
                self.F[tuple(simplex)] = 1
            elif len(simplex) == 2:
                u, v = simplex
                # edge weight
                self.F[tuple(simplex)] = self.adj[u][v]['weight'] if self.adj.has_edge(u, v) else 1e-7
            elif len(simplex) == 3:
                u, v, w = simplex
                if self.adj.has_edge(u, v) and self.adj.has_edge(u, w) and self.adj.has_edge(v, w):
                    # geometric mean of edge weights
                    self.F[tuple(simplex)] = (
                    self.adj[u][v]['weight'] * self.adj[u][w]['weight'] * self.adj[v][w]['weight']
                    )**(1/3)
                else:
                    self.F[tuple(simplex)] = float('inf') 
    
    def build_simplicial_pair(self):
        radii = self.radii
        self.value_list = []
        for r in radii:
            self.value_list.append([r, r + 0])
        #print(self.value_list)
        # build dictionary of simplices that will be used for the calculation of coboundary matrices
        edge_idx = 0
        face_idx = 0
        self.C_0 = []
        self.C_1,  self.C_2 = {}, {}
        self.fil_1, self.fil_2 = [], [] # store filtration values, will be converted to numpy arrays.
        for simplex, filtration in self.simplex_tree.get_filtration():
            if filtration >= self.value_list[-1][-1]:
                break    
            if len(simplex) == 1:
                self.C_0.append(simplex)
            if len(simplex) == 2:
                self.fil_1.append(filtration)
                self.C_1[tuple(simplex)] = (edge_idx, filtration) 
                edge_idx += 1
            if len(simplex) == 3:
                self.fil_2.append(filtration)
                self.C_2[tuple(simplex)] = (face_idx, filtration)
                face_idx += 1
        self.fil_1, self.fil_2 = np.array(self.fil_1), np.array(self.fil_2)

    def build_matrices(self):
        self.d_0 = coboundary_nonconstant_0(self.C_0, self.C_1, self.charges, self.F)
        self.d_1 = coboundary_nonconstant_1(self.C_1, self.C_2, self.charges, self.F)

    def psl_0(self): 
        res = [] 
        for _, v1 in self.value_list:     
            d_0_tp = self.d_0[:sum(self.fil_1<=v1)]
            res.append(np.dot(d_0_tp.T, d_0_tp)) 
        return res

    def psl_1(self):
        res = []
        for v0, v1 in self.value_list:
            d_0_t = self.d_0[:sum(self.fil_1<=v0)]
            d_1_tp = self.d_1[:sum(self.fil_2<=v1), :sum(self.fil_1<=v1)]
            if sum(self.fil_1<=v0) == sum(self.fil_1<=v1):
                res.append(np.dot(d_0_t, d_0_t.T) + np.dot(d_1_tp.T, d_1_tp)) 
            else:
                tmp = np.dot(d_1_tp.T, d_1_tp)
                tmp_idx = sum(self.fil_1<=v0)
                A, B, C, D = tmp[:tmp_idx, :tmp_idx], tmp[:tmp_idx, tmp_idx:], tmp[tmp_idx:, :tmp_idx],tmp[tmp_idx:, tmp_idx:]  
                res.append(np.dot(d_0_t, d_0_t.T) + A - B@np.linalg.pinv(D)@C)
        return res
