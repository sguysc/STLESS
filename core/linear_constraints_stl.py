import numpy as np
from itertools import product
# import polygons
from shapely.ops import triangulate # poorman's convex decomposition
import cdd # to convert V-repr polygons to H-repr
import copy
# import  mtl



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
    def __init__(self, A, b, name, i_constr):
        """
        Defines linear functions f(x) = Ax + b.
        The integration domain is defined as the union of where all of these functions are positive if mode='Union'
        or the domain where any of the functions is positive, when mode='Intersection'
        :param A: matrix A with shape (MxL, D) where M is the number of constraints and D the dimension, L is the number of convex polytopes it is made of
        :param b: offset, shape (MxL, 1)
        """
        self.A = A
        self.b = b
        self.name = name
        self.N_constraints = b.shape[0]
        self.N_dim = A.shape[1]
        self.i_constr = i_constr

    def evaluate(self, x):
        """
        Evaluate linear functions at N locations x
        :param x: location, shape (D, N)
        :return: Ax + b
        """
        # every evaluation of the point with all the linear domains
        sol = np.dot(self.A, x) + self.b

        per_domain = np.split(sol, self.i_constr, axis=0)
        # get the min value of each domain, that would be its respected "robustness"
        # but the final robustness would be the maximun of those mins. why? because if
        # we have two regions and the point is within one but not the other, then one
        # will have a positive min, the other will have a negative min but the overall
        # solution is the positive robustness because essentially it is within the domain.
        # filled gets it back from a maskedarray to a regular one
        min_R_per_domain = np.array([np.min(d, axis=0) for d in per_domain])

        return np.max(min_R_per_domain, axis=0)

    def robustness(self, x):
        # because it is basically just that
        return self.evaluate(x)

    def integration_domain(self, x):
        """
        is 1 if x is in the integration domain, else 0
        :param x: location, shape (D, N)
        :return: either self.indicator_union or self.indicator_intersection, depending on setting of self.mode
        """
        return self.indicator_intersection(x)

    def indicator_intersection(self, x):
        """
        Intersection of indicator functions taken to be 1 when the linear function is >= 0
        :param x: location, shape (D, N)
        :return: 1 if all linear functions are >= 0, else 0.
        """
        return np.where(self.evaluate(x) >= 0, 1, 0)



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



# GUY: added new class to store a bunch of polydyra
class MultiLinearConstraints():
    def __init__(self, predicates, specification, n, dim, spec_pos=True, Rot=None, Trans=None, extra_points_to_check=None):

        self.horizon = n
        self.dim = dim
        self.N_dim = dim * n

        self.specification = copy.deepcopy( specification )

        if(Rot is None):
            self.Rot = np.eye(self.N_dim)
        else:
            self.Rot = Rot.copy()
        if(Trans is None):
            self.Trans = np.zeros((self.N_dim, 1))
        else:
            self.Trans = Trans.copy()

        self.extra_points = extra_points_to_check

        # where we store each predicate
        self.domains = []
        self.sign = -1.0 if spec_pos else 1.0
        self.time_list = []

        Ai = np.empty( (0, self.N_dim) )
        bi = np.empty( (0, 1) )
        first_pred = True
        for prop, prop_values in predicates.items():
            A, b = prop_values['A'], prop_values['b']
            i_constr  = np.cumsum(prop_values['split'][:-1])
            self.domains.append( LinearConstraints(A, b, prop, i_constr) )
            xtra_i = 0

            for t in range(n):
                Atemp = np.zeros( (b.shape[0], self.N_dim) )
                Atemp[:, t*dim:(t+1)*dim] = A

                Ai = np.vstack((Ai, Atemp))
                bi = np.vstack((bi, b))

                if(first_pred):
                    self.time_list.append(t)

                if(self.extra_points):
                    if(xtra_i < len(self.extra_points)):
                        init_pnt, end_pnt, ratio = self.extra_points[xtra_i]
                        if(t == init_pnt):
                            Atemp = np.zeros( (b.shape[0], self.N_dim) )
                            Atemp[:, init_pnt*dim:(init_pnt+1)*dim] = ratio*A
                            Atemp[:, end_pnt*dim:(end_pnt+1)*dim] = (1.0-ratio)*A
                            # self.extra_points.pop(0)
                            xtra_i += 1
                            Ai = np.vstack((Ai, Atemp))
                            bi = np.vstack((bi, b))

                            if(first_pred):
                                self.time_list.append(init_pnt + ratio*(end_pnt-init_pnt))

            first_pred = False
        # transform it via the x=Ly+c to y~N(0,I)
        #Ax+b>0 => ALy+Ac+b>0 => A'y+b'>0
        bp = Ai @ self.Trans + bi
        Ap = Ai @ self.Rot
        # store for later use when we add a shift
        self._all_hyperplanes_A = Ap.copy()
        self._all_hyperplanes_b = bp.copy()
        self.A = Ap.copy()
        self.b = bp.copy()

        # to be able to differentiate this from the old models
        self.stl = True
        self.shift_val = 0.


    def robustness(self, xtrajs):
        """
        Find the robustness metric for all predicates at N locations x
        :param x: location, shape (D, N)
        :return: rob(Pred(i)) for all i \in domains
        """
        # breakpoint()
        # rtamt formulation
        # t_vec = np.arange(0, self.horizon)
        t_vec = self.time_list
        num_trajs = xtrajs.shape[1]
        rob = np.zeros((num_trajs))
        # we move this back to the original coordinates because the predicates expect
        # it in the original coordinates. otherwise, we would have needed a predicate
        # for every time step
        rot_xtrajs = self.Rot @ xtrajs + self.Trans
        for i in range(num_trajs):
            # GUY: there's room for improvement here, if you take all of the trajs
            # together and send it to the "domain" and then split it for each robustness.
            # breakpoint()
            xt = rot_xtrajs[:,i:i+1]
            all_states = xt.reshape((-1,self.dim)).T
            if(self.extra_points):
                for p_i,p_e,ratio in reversed(self.extra_points):
                    mean_state = all_states[:,p_i:p_i+1] + ratio*(all_states[:,p_e:p_e+1]-all_states[:,p_i:p_i+1])
                    all_states = np.insert(all_states, p_i+1, mean_state.T, axis=1)
            trajectory = {'time': t_vec} #.tolist()
            for predicate in self.domains:
                rob_per_pred = predicate.evaluate(all_states)
                trajectory.update({str(predicate.name): rob_per_pred.tolist()})
            # there's some bug in the rtamt library that it gives wrong results on the first
            # go, and then it's good on the second time you call it. not sure what's the problem.
            # we could also call "parse" again, that works, but I guessed it would be slower.
            # GUY TODO: if there's time, try to debug and see why this happens.
            # _rob = self.specification.evaluate(trajectory)
            # this is doing the actual STL
            rob_traj = self.specification.evaluate(trajectory)
            # breakpoint()
            # rob[i] = rob_traj[-1][1] # the last value, is the signal's robustness. GUY: CHECK, it used to be different in mtl library
            # We define the robustness degree rho(phi,w) as rho(phi,w,0)
            rob[i] = rob_traj[0][1] # search the min of all steps to see if it failed
        # mtl formulation
        # t_vec = np.arange(0, self.horizon)
        # num_trajs = xtrajs.shape[1]
        # rob = np.zeros((num_trajs))
        # # we move this back to the original coordinates because the predicates expect
        # # it in the original coordinates. otherwise, we would have needed a predicate
        # # for every time step
        # rot_xtrajs = self.Rot @ xtrajs + self.Trans
        # for i in range(num_trajs):
        #     # GUY: there's room for improvement here, if you take all of the trajs
        #     # together and send it to the "domain" and then split it for each robustness.
        #     xt = rot_xtrajs[:,i:i+1]
        #     trajectory = {}
        #     for predicate in self.domains:
        #         all_states = xt.reshape((-1,self.dim)).T
        #         rob_per_pred = predicate.evaluate(all_states)

        #         signal = zip(t_vec, rob_per_pred)
        #         trajectory.update({str(predicate.name): [step for step in signal]})
        #     # this is doing the actual STL
        #     rob[i] = self.specification(trajectory) #, dt=1)
        return rob


    def evaluate(self, x):
        """
        Evaluate linear functions at N locations x
        :param x: location, shape (D, N)
        :return: Ax + b
        """
        breakpoint()
        vals = np.dot(self.A, x) + self.b # - self.sign * self.shift_val

        return vals

    def integration_domain(self, x):
        """
        is 1 if x is in the integration domain, else 0
        :param x: location, shape (D, N)
        :return: either self.indicator_union
        """
        # breakpoint()
        # now we check it by robustness
        in_domain = np.where(self.robustness(x) >= self.shift_val, 1, 0)

        return in_domain

    def shift(self, shft):
        # it's minus because we calculate the shift with a minus (as in the old version)
        # a positive shft means the robustness is less than zero (failing)
        self.shift_val = -shft
        if(np.abs(self.shift_val) > 0.0):
            # since we don't know what is positive or negative predicates, we will
            # have twice as many hyperplanes right now for + and -.
            self.A = np.vstack((self._all_hyperplanes_A     , self._all_hyperplanes_A))
            self.b = np.vstack((self._all_hyperplanes_b+shft, self._all_hyperplanes_b-shft))
        else:
            # no reason to double the hyperplanes if shift is zero (last nesting)
            self.A = self._all_hyperplanes_A.copy()
            self.b = self._all_hyperplanes_b.copy()


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

