import numpy as np
import time

from .nestings import HDRNesting
from .integration_tracker import HDRTracker
from .integration_loop import IntegrationLoop


class HDR(IntegrationLoop):
    def __init__(self, linear_constraints, shift_sequence, n_samples=64, X_init=None, \
                 n_skip=0, timing=False, std_des=0.005):
        """
        Holmes-Diaconis-Ross algorithm for estimating integrals of linearly constrained Gaussians
        :param linear_constraints: instance of LinearConstraints
        :param shift_sequence: sequence of numbers > 0 that define the nestings (e.g. shifts from subset simulation)
        :param n_samples: number of samples per nesting (integer)
        :param X_init: starting points for ESS, the ith column has to be in the ith nesting
        :param n_skip: number of samples to skip in ESS
        :param timing: whether to measure the runtime
        """
        super().__init__(linear_constraints, n_samples, n_skip)

        self.shift_sequence = shift_sequence
        self.X_init = X_init
        self.tracker = HDRTracker(self.shift_sequence)

        self.std_evaluation1 = 1.
        self.std_evaluation2 = 1.
        self.std = 1.0

        # timing of every iteration in the core
        self.timing = timing
        if self.timing:
            self.times = []

        # adaptive number of samples per nesting (assuming p=0.5)
        if(n_samples is None):
            #  ---> number of nestings
            #|
            #V  number of points in a nest
            std_vals = np.array([ \
            [0.2500,    0.1875,    0.1220,    0.0750,    0.0448,    0.0262,    0.0152,    0.0087,     0.0050,    0.0028],\
            [0.1768,    0.1288,    0.0814,    0.0485,    0.0280,    0.0158,    0.0088,    0.0049,     0.0027,    0.0015],\
            [0.1250,    0.0898,    0.0558,    0.0327,    0.0186,    0.0103,    0.0057,    0.0031,     0.0017,    0.0009],\
            [0.0884,    0.0630,    0.0389,    0.0226,    0.0127,    0.0070,    0.0038,    0.0021,     0.0011,    0.0006],\
            [0.0625,    0.0444,    0.0273,    0.0158,    0.0089,    0.0049,    0.0026,    0.0014,     0.0008,    0.0004],\
            [0.0442,    0.0313,    0.0192,    0.0111,    0.0062,    0.0034,    0.0018,    0.0010,     0.0005,    0.0003],\
            [0.0312,    0.0221,    0.0136,    0.0078,    0.0044,    0.0024,    0.0013,    0.0007,     0.0004,    0.0002],\
            [0.0221,    0.0156,    0.0096,    0.0055,    0.0031,    0.0017,    0.0009,    0.0005,     0.0003,    0.0001],\
            [0.0156,    0.0111,    0.0068,    0.0039,    0.0022,    0.0012,    0.0006,    0.0003,     0.0002,    0.0001],\
            ])
            n_vals= np.array([4,8,16,32,64,128,256,512,1024])
            n_nest = 10 if len(shift_sequence) > 10 else  len(shift_sequence)
            # breakpoint()
            idx = self.find_nearest(std_vals[:,n_nest], std_des)
            self.n_samples = n_vals[idx]
            print('adaptive n=%d to aim for sigma=%.3f' %(self.n_samples, std_des))

    def run(self, verbose=False):
        """
        Run the HDR method
        :return:
        """
        # breakpoint()
        for i, shift in enumerate(self.shift_sequence):
            if self.timing:
                t = time.process_time()

            if i == 0:
                X = np.random.randn(self.dim, self.n_samples)
            else:
                X = current_nesting.sample_from_nesting(self.n_samples, self.X_init[:, i, None], self.n_skip)

            current_nesting = HDRNesting(self.lincon, shift)
            current_nesting.compute_log_nesting_factor(X)
            # compute the variance/standard deviation. see paper for details
            # var = prod( p*(1-p)/n + p**2) - prod(p**2)
            p = np.exp(current_nesting.log_conditional_probability)
            n = X.shape[1]
            # add (n, p)
            self.std_evaluation1 *= (p*(1.0-p)/n + p**2)
            self.std_evaluation2 *= (p**2)
            self.tracker.add_nesting(current_nesting)

            if self.timing:
                self.times.append(time.process_time() - t)
            if verbose:
                print('finished nesting #{}'.format(i))

        self.std = np.sqrt( self.std_evaluation1 - self.std_evaluation2 )


        # saving the samples from the domain of interest
        self.tracker.add_samples(X[:, self.lincon.integration_domain(X)==1])

    def draw_from_domain(self, n):
        """
        Sample from the domain of interest.
        :param n: number of samples to draw
        :return: samples (D, n)
        """
        domain = HDRNesting(self.lincon, 0.)
        return domain.sample_from_nesting(n, self.X_init[:, -1, None], self.n_skip)

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx