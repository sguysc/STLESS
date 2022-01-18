import numpy as np
from itertools import product
# import polygons
from shapely.ops import triangulate # poorman's convex decomposition
import cdd # to convert V-repr polygons to H-repr

def convex_decompose(poly_list):
    # pa = poly_list[0]
    out_regions = []
    for n, pa in enumerate( poly_list ):
        try:
            # first, lets check, maybe it is already convex so we can return it back
            r_v = np.array(pa.boundary.xy).T
            if(  is_polygon_convex(r_v[:-1,:]) ):
                dct = __get_A_b(r_v)
                out_regions.append( dct )
                continue
        except:
            # if we're here, it means that it is a complex shape with holes, so obviously not convex
            pass

        regions = triangulate(pa)
        pa_decomp = None
        tmp_region = None
        i=0
        for r in regions:
            # breakpoint()
            i +=1
            # some preprocessing to take care of triangles that are outside of the region
            # or inside and outside
            if( not pa.contains(r) ):
                # at least some portion of the region is outside, but maybe not all of it
                if(pa.intersection(r).area > 0.0):
                    # this would be the new region to check
                    r = pa.intersection(r)

            if( pa.contains(r) ):
                #only deal with the triangles inside the domain
                if(pa_decomp is None):
                    pa_decomp = r
                else:
                    pa_decomp = pa_decomp.union(r)

                if(tmp_region is None):
                    tmp_region = r
                    # one region is a triangle so it is always convex, move to the next
                    continue
                else:
                    # this is greedy approach to try to add some triangles together
                    tmp_region_ext = tmp_region.union(r)
                    # breakpoint()
                    try:
                        r_v = np.array(tmp_region_ext.boundary.xy).T
                        if(  is_polygon_convex(r_v[:-1,:]) ):
                            tmp_region = tmp_region_ext
                            continue
                        else:
                            # it is within the original shape, so store it
                            r_v=np.array(tmp_region.boundary.xy).T
                            dct = __get_A_b(r_v)
                            out_regions.append( dct )
                            tmp_region = r # reset
                    except:
                        # this happens when the shapes are not connected -> multipolygon
                        r_v=np.array(tmp_region.boundary.xy).T
                        dct = __get_A_b(r_v)
                        out_regions.append( dct )
                        tmp_region = r # reset

        # deal with the last one
        r_v=np.array(tmp_region.boundary.xy).T
        dct = __get_A_b(r_v)
        out_regions.append( dct )

        # just a check that we didn't miss any areas
        pa_diff = pa - pa_decomp
        assert pa_diff.area == 0.0, 'sum of the convex areas does not match original polygon (poly: %d)' %n

    return out_regions

def __get_A_b(r_v):
    # this is the format that cdd expects to get it
    Vformat = np.hstack( ( np.ones((r_v.shape[0],1)), r_v ) )
    mat     = cdd.Matrix(Vformat, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    V_poly  = cdd.Polyhedron(mat)
    H_poly  = np.array( V_poly.get_inequalities() )
    b       = H_poly[:,0:1]
    A       = H_poly[:,1:]
    return {'A': A, 'b': b}


def is_polygon_convex(Points):
    # For each set of three adjacent points A, B, C,
    # find the cross product AB · BC. If the sign of
    # all the cross products is the same, the angles
    # are all positive or negative (depending on the
    # order in which we visit them) so the polygon is convex.
    got_negative = False
    got_positive = False
    num_points = len(Points)
    for A in range(num_points):
        B = (A + 1) % num_points
        C = (B + 1) % num_points

        cross_product = cross_product_length( \
                Points[A][0], Points[A][1], \
                Points[B][0], Points[B][1], \
                Points[C][0], Points[C][1])
        if (cross_product < 0):
            got_negative = True
        elif (cross_product > 0):
            got_positive = True
        if (got_negative and got_positive):
            return False

    # If we got this far, the polygon is convex.
    return True

def cross_product_length(Ax, Ay, Bx, By, Cx, Cy):
    # Get the vectors' coordinates.
    BAx = Ax - Bx
    BAy = Ay - By
    BCx = Cx - Bx
    BCy = Cy - By

    # Calculate the Z coordinate of the cross product.
    return (BAx * BCy - BAy * BCx)

class LinearConstraints():
    def __init__(self, A, b, mode='Intersection'):
        """
        Defines linear functions f(x) = Ax + b.
        The integration domain is defined as the union of where all of these functions are positive if mode='Union'
        or the domain where any of the functions is positive, when mode='Intersection'
        :param A: matrix A with shape (M, D) where M is the number of constraints and D the dimension
        :param b: offset, shape (M, 1)
        """
        self.A = A
        self.b = b
        self.N_constraints = b.shape[0]
        self.N_dim = A.shape[1]
        self.mode = mode
        if(self.mode == 'Complement'):
            self.inout = -1.
        else:
            self.inout =  1.
        # breakpoint()

    def evaluate(self, x):
        """
        Evaluate linear functions at N locations x
        :param x: location, shape (D, N)
        :return: Ax + b
        """
        return self.inout * ( np.dot(self.A, x) + self.b )

    def integration_domain(self, x):
        """
        is 1 if x is in the integration domain, else 0
        :param x: location, shape (D, N)
        :return: either self.indicator_union or self.indicator_intersection, depending on setting of self.mode
        """
        if self.mode == 'Union':
            return self.indicator_union(x)
        elif self.mode == 'Intersection':
            return self.indicator_intersection(x)
        elif self.mode == 'Complement':
            return self.indicator_complement(x)
        else:
            raise NotImplementedError

    def indicator_intersection(self, x):
        """
        Intersection of indicator functions taken to be 1 when the linear function is >= 0
        :param x: location, shape (D, N)
        :return: 1 if all linear functions are >= 0, else 0.
        """
        return np.where(self.evaluate(x) >= 0, 1, 0).prod(axis=0)

    def indicator_union(self, x):
        """
        Union of indicator functions taken to be 1 when the linear function is >= 0
        :param x: location, shape (D, N)
        :return: 1 if any of the linear functions is >= 0, else 0.
        """
        return 1 - (np.where(self.evaluate(x) >= 0, 0, 1)).prod(axis=0)

    def indicator_complement(self, x):
        """
        Intersection of indicator functions taken to be 1 when the linear function is <= 0
        :param x: location, shape (D, N)
        :return: 1 if all linear functions are <= 0, else 0.
        """
        return np.where(self.evaluate(x) <= 0, 1, 0).prod(axis=0)


# class MyLinearConstraints():
#     def __init__(self, A, b, mode='Intersection'):
#         """
#         Defines linear functions f(x) = Ax + b.
#         The integration domain is defined as the union of where all of these functions are positive if mode='Union'
#         or the domain where any of the functions is positive, when mode='Intersection'
#         :param A: matrix A with shape (M, D) where M is the number of constraints and D the dimension
#         :param b: offset, shape (M, 1)
#         """
#         self.A = A
#         self.b = b
#         self.N_constraints = b.shape[0]
#         self.N_dim = A.shape[1]
#         self.mode = mode
#         if(self.mode == 'Complement'):
#             self.inout = -1.
#         else:
#             self.inout =  1.
#         # breakpoint()

#     def evaluate(self, x):
#         """
#         Evaluate linear functions at N locations x
#         :param x: location, shape (D, N)
#         :return: Ax + b
#         """
#         return self.inout * ( np.dot(self.A, x) + self.b )

#     def integration_domain(self, x):
#         """
#         is 1 if x is in the integration domain, else 0
#         :param x: location, shape (D, N)
#         :return: either self.indicator_union or self.indicator_intersection, depending on setting of self.mode
#         """
#         if self.mode == 'Union':
#             return self.indicator_union(x)
#         elif self.mode == 'Intersection':
#             return self.indicator_intersection(x)
#         elif self.mode == 'Complement':
#             return self.indicator_complement(x)
#         else:
#             raise NotImplementedError

#     def indicator_intersection(self, x):
#         """
#         Intersection of indicator functions taken to be 1 when the linear function is >= 0
#         :param x: location, shape (D, N)
#         :return: 1 if all linear functions are >= 0, else 0.
#         """
#         return np.where(self.evaluate(x) >= 0, 1, 0).prod(axis=0)

#     def indicator_union(self, x):
#         """
#         Union of indicator functions taken to be 1 when the linear function is >= 0
#         :param x: location, shape (D, N)
#         :return: 1 if any of the linear functions is >= 0, else 0.
#         """
#         return 1 - (np.where(self.evaluate(x) >= 0, 0, 1)).prod(axis=0)

#     def indicator_complement(self, x):
#         """
#         Intersection of indicator functions taken to be 1 when the linear function is <= 0
#         :param x: location, shape (D, N)
#         :return: 1 if all linear functions are <= 0, else 0.
#         """
#         return np.where(self.evaluate(x) <= 0, 1, 0).prod(axis=0)



class ShiftedLinearConstraints(LinearConstraints):
    def __init__(self, A, b, shift):
        """
        Class for shifted linear constraints that appear in multilevel splitting method
        :param A: matrix A with shape (M, D) where M is the number of constraints and D the dimension
        :param b: offset, shape (M, 1)
        :param shift: (positive) scalar value denoting the shift
        """
        self.shift = shift
        super(ShiftedLinearConstraints, self).__init__(A, b + shift)


# # GUY: added new class to store a bunch of polydyra
# class ListMultiLinearConstraints():
#     def __init__(self, list_multilincon):

#         self.list_multilincon = list_multilincon

#     def evaluate(self, x):
#         for con in self.list_multilincon:
#             con.evaluate(x)

#     def integration_domain(self, x):
#         for con in self.list_multilincon:
#             con.integration_domain(x)

#     def indicator_union(self, x):
#         for con in self.list_multilincon:
#             con.indicator_union(x)


# GUY: added new class to store a bunch of polydyra
class MultiLinearConstraints():
    def __init__(self, intersection_constraints, n, dim, Rot=None, Trans=None, extra_points=None):
        '''
        Parameters
        ----------
        A : matrix shape (M,D) where M is the number of constraints and D the dimension
            of the constraint in the configuration space.
        b : offset, shape (M, 1) (in the configuration space.)
        n : int, number of repetitions of the config space (ie, the horizon).
        extra_points: array of tuples of (initial point, end point, ratio between them)
        '''
        self.horizon = n
        self.dim = dim
        self.N_dim = dim * n
        self.intersection_constraints = intersection_constraints.copy()
        self.N_constraints = np.empty( (0) )

        if(Rot is None):
            self.Rot = np.eye(self.N_dim)
        else:
            self.Rot = Rot.copy()
        if(Trans is None):
            self.Trans = np.zeros((self.N_dim, 1))
        else:
            self.Trans = Trans.copy()

        # store all the regions, each of them is an intersection of all its constraints
        self.polyhydras = []
        inits, goals, goals_c, obstacles, obstacles_c = [], [], [], [], []
        inits_dict, goals_dict, goals_c_dict, obs_dict, obs_c_dict = {}, {}, {}, {}, {}
        obs_ids = []
        # store it encoded so we can quickly call the permutations later
        self.constr_dict = {}
        for C in intersection_constraints:
            typ, nid = C['type'], C['id']
            A, b, t0, t1, grp = C['A'], C['b'], C['t0'], C['t1'], C['grp']
            if(typ == 'o'):
                for t in range(t0,t1+1):
                    # obstacles.append('o_%d_%d' %(nid, t))
                    prev_val = obs_dict.get(grp)
                    if(prev_val is None):
                        obs_dict.update( {grp: ['o_%d_%d' %(nid, t)]})
                    else:
                        obs_dict.update( {grp: prev_val + ['o_%d_%d' %(nid, t)]})

                    self.constr_dict['o_%d_%d' %(nid, t)] = {'A':A, 'b':b, 't0':t0, 't1':t1, 'grp':grp}
                obs_ids.append(nid)
            if(typ == 'oc'):
                # print('did not implemented this obs_comp')
                # obstacles_c.append(['oc_%d_%d' %(nid, t) for t in range(t0,t1+1)])
                for t in range(t0,t1+1):
                    prev_val = obs_c_dict.get(grp)
                    if(prev_val is None):
                        obs_c_dict.update( {grp: ['oc_%d_%d' %(nid, t)]})
                    else:
                        obs_c_dict.update( {grp: prev_val + ['oc_%d_%d' %(nid, t)]})

                    self.constr_dict['oc_%d_%d' %(nid, t)] = {'A':A, 'b':b, 't0':t0, 't1':t1, 'grp':grp}
            if(typ == 'i'):
                for t in range(t0,t1+1):
                    # inits.append('i_%d_%d' %(nid, t))
                    prev_val = inits_dict.get(grp)
                    if(prev_val is None):
                        inits_dict.update( {grp: ['i_%d_%d' %(nid, t)]})
                    else:
                        inits_dict.update( {grp: prev_val + ['i_%d_%d' %(nid, t)]})

                    self.constr_dict['i_%d_%d' %(nid, t)] = {'A':A, 'b':b, 't0':t0, 't1':t1, 'grp':grp}
            if(typ == 'g'):
                for t in range(t0,t1+1):
                    # goals.append('g_%d_%d' %(nid, t))
                    prev_val = goals_dict.get(grp)
                    if(prev_val is None):
                        goals_dict.update( {grp: ['g_%d_%d' %(nid, t)]})
                    else:
                        goals_dict.update( {grp: prev_val + ['g_%d_%d' %(nid, t)]})

                    self.constr_dict['g_%d_%d' %(nid, t)] = {'A':A, 'b':b, 't0':t0, 't1':t1, 'grp':grp}
            if(typ == 'gc'):
                for t in range(t0,t1+1):
                    # goals_c.append('gc_%d_%d' %(nid, t))
                    prev_val = goals_c_dict.get('%d_%d'%(grp,t))
                    if(prev_val is None):
                        goals_c_dict.update( {'%d_%d'%(grp,t): ['gc_%d_%d' %(nid, t)]})
                    else:
                        goals_c_dict.update( {'%d_%d'%(grp,t): prev_val + ['gc_%d_%d' %(nid, t)]})

                    self.constr_dict['gc_%d_%d' %(nid, t)] = {'A':A, 'b':b, 't0':t0, 't1':t1, 'grp':grp}
        # create the disjoint regions
        # so, the user must specify the correct group number according to whether they
        # need it to be 'or' or 'and'. for example, the spec: G[!O1] ^( F[G1] v F[G2] )
        # when constructing traj1, we want to hit O1 and get in either G1 or G2, which mean
        # that they behave like the same obstacle, even if they are also at different times.
        # so you give it the same grp number. but when doing traj23, you want it being both
        # in !G1 and in !G2 at the same time (traj, not nec. t) so you give it different
        # grp number so they will create the permutation between themselves. not only that,
        # since they could also span on different times, it has to take all the free
        # regions at all times too (because the original spec is "or"). so (!G1_t0 ^ !G1_t1
        # ^ !G2_t3) v ...
        # breakpoint()
        disj_rgns = []
        for k,v in obs_dict.items():
            disj_rgns.append( v )
        for k,v in obs_c_dict.items():
            disj_rgns.append( v )
        for k,v in inits_dict.items():
            disj_rgns.append( v )
        for k,v in goals_dict.items():
            disj_rgns.append( v )
        for k,v in goals_c_dict.items():
            disj_rgns.append( v )

        list1 = list(product(*disj_rgns))
        # holds all the possible unions
        # breakpoint()
        # if(len(inits) == 0):
        #     inits = ['empty']
        # if(len(goals) == 0):
        #     if(len(goals_c)>0):
        #         goals = goals_c.copy()
        #     else:
        #         goals = ['empty']
        # if(len(obstacles) == 0):
        #     obstacles = ['empty']
        #     if(len(obstacles_c) == 0):
        #         obstacles_c = ['empty']
        #         # still send this, because we deal with the empty sets
        #         list1 = list(product(inits, goals, obstacles))
        #     else:
        #         # this will transpose so it will be in the right notation for the product
        #         obstacles_c = [list(x) for x in zip(*obstacles_c)]
        #         # we're dealing with complimentary shapes here
        #         list1 = list(product(inits, goals, *obstacles_c))
        # else:
        #     # then it's probably traj type1 and we do perm(init,goal,obs)
        #     list1 = list(product(inits, goals, obstacles))

        # breakpoint()
        # create the actual linconstraints
        for perm in list1:
            Ai = np.empty( (0, self.N_dim) )
            bi = np.empty( (0, 1) )
            #we assume that goals and obstacles are disjoint this means that
            #we can't have obstacle at time t and goal at time t at the same time.
            #it's not that the lincongauss won't handle it, but it would just waste
            # a lot of time figuring out that the probability of that is zero.
            valid_setting = True
            obs_t, goal_t = -1,-2

            for poly_cnstr in perm:
                if('empty' in poly_cnstr):
                    continue
                A, b = self.constr_dict[poly_cnstr]['A'], self.constr_dict[poly_cnstr]['b']
                t = int(poly_cnstr.split('_')[-1])
                typ = poly_cnstr.split('_')[0]
                if('g' in typ):
                    goal_t = t
                elif('o' in typ):
                    obs_t = t

                if(obs_t == goal_t):
                    # occupying two disjoint spaces at the same time
                    valid_setting = False
                    break
                Atemp = np.zeros( (b.shape[0], self.N_dim) )
                Atemp[:, t*dim:(t+1)*dim] = A

                # transform it via the x=Ly+c to y~N(0,I)
                # Ax+b>0 => ALy+Ac+b>0 => A'y+b'>0
                b = Atemp @ self.Trans + b
                Atemp = Atemp @ self.Rot

                Ai = np.vstack((Ai, Atemp))
                bi = np.vstack((bi, b))
            if(valid_setting):
                # that's a new permutation of the total union, so add a constraint for it
                self.polyhydras.append( LinearConstraints(Ai, bi) )
                self.N_constraints = np.append(self.N_constraints, bi.shape[0])

        if(extra_points):
            # breakpoint()
            # here we do it only for the obstacles
            for init_pnt, end_pnt, ratio in extra_points:
                for obs_id in obs_ids:
                    perm_i = [item for item in list1 if 'o_%d_%d'%(obs_id, init_pnt) in item][0]
                    perm_e = [item for item in list1 if 'o_%d_%d'%(obs_id, end_pnt) in item][0]

                    Ai = np.empty( (0, self.N_dim) )
                    bi = np.empty( (0, 1) )

                    Aobs_i, bobs_i = None, None

                    for poly_cnstr in perm_i:
                        if('empty' in poly_cnstr):
                            continue
                        A, b = self.constr_dict[poly_cnstr]['A'], self.constr_dict[poly_cnstr]['b']
                        if('o' in poly_cnstr):
                            A = A*ratio # to account for the "mid"-point
                            b = b*ratio

                        t = int(poly_cnstr.split('_')[-1])
                        Atemp = np.zeros( (b.shape[0], self.N_dim) )
                        Atemp[:, t*dim:(t+1)*dim] = A

                        if('o' in poly_cnstr):
                            Aobs_i = Atemp.copy()
                            bobs_i = b.copy()
                        else:
                            # transform it via the x=Ly+c to y~N(0,I)
                            # Ax+b>0 => ALy+Ac+b>0 => A'y+b'>0
                            b = Atemp @ self.Trans + b
                            Atemp = Atemp @ self.Rot
                            Ai = np.vstack((Ai, Atemp))
                            bi = np.vstack((bi, b))

                    for poly_cnstr in perm_e:
                        if('o' not in poly_cnstr):
                            continue

                        A, b = self.constr_dict[poly_cnstr]['A'], self.constr_dict[poly_cnstr]['b']
                        A = A*(1.0 - ratio) # to account for the "mid"-point
                        b = b*(1.0 - ratio)

                        t = int(poly_cnstr.split('_')[-1])
                        # Atemp = np.zeros( (b.shape[0], self.N_dim) )
                        Aobs_i[:, t*dim:(t+1)*dim] = A
                        b += bobs_i

                        # transform it via the x=Ly+c to y~N(0,I)
                        # Ax+b>0 => ALy+Ac+b>0 => A'y+b'>0
                        b = Aobs_i @ self.Trans + b
                        Aobs_i = Aobs_i @ self.Rot

                        Ai = np.vstack((Ai, Aobs_i))
                        bi = np.vstack((bi, b))
                    # that's a new permutation of the total union, so add a constraint for it
                    self.polyhydras.append( LinearConstraints(Ai, bi) )
                    self.N_constraints = np.append(self.N_constraints, bi.shape[0])


    def evaluate(self, x):
        """
        Evaluate linear functions at N locations x
        :param x: location, shape (D, N)
        :return: Ax + b
        """
        vals = None
        for P in self.polyhydras:
            val = P.evaluate(x)
            if(vals is None):
                vals = val.copy()
            else:
                vals = np.append(vals,val,axis=0)

        return vals

    def integration_domain(self, x):
        """
        is 1 if x is in the integration domain, else 0
        :param x: location, shape (D, N)
        :return: either self.indicator_union
        """
        # breakpoint()
        return self.indicator_union(x)

    def indicator_union(self, x):
        """
        Union of indicator functions taken to be 1 when the linear function is >= 0
        :param x: location, shape (D, N)
        :return: 1 if ANY of the linear functions is >= 0, else 0.
        """
        N = x.shape[-1]
        in_domain = np.zeros(N) # len(self.polyhydras)
        # cnt = 0
        for j in range(N):
            for i, P in enumerate(self.polyhydras):
                ret = P.integration_domain(x[:,j:j+1])
                if(ret):
                    # we got some intersections
                    in_domain[j] = ret
                # GUY: check if this is correct, or should be any.
                # is all of the points of one polytope
                # if(ret.any()):
                #     in_domain[i] = 1

        return in_domain

    def shift(self, shift):
        # use this instead of the shiftedMulti because it is faster than reconstructing everything
        for P in self.polyhydras:
            P.b += shift

class ShiftedMultiLinearConstraints(MultiLinearConstraints):
    def __init__(self, intersection_constraints, n, dim, shift):
        """
        Class for shifted linear constraints that appear in multilevel splitting method
        :param A: matrix A with shape (M, D) where M is the number of constraints and D the dimension
        :param b: offset, shape (M, 1)
        :param shift: (positive) scalar value denoting the shift
        """
        self.shift = shift
        # GUY: need to add the shift somehow
        print('do not forget to add the shift!!')
        super(ShiftedMultiLinearConstraints, self).__init__(intersection_constraints, n, dim)

