import numpy as np
from flecks.util.kron_util import kron_mat_vec_prod, kron_multi

class KronKernel:
    """ maintains a list of kernels corresponding to each dimension. 
    This class assumes that the Kernel over the multi-dimensional space
    is a tensor product kernel, so it's Gram matrix can only be instantiated 
    over *regular grids* in each dimension.  This is ideal for a discretized 
    approximation to a space.
    """

    def __init__(self, kernel_list=None):
        self._kerns = kernel_list
        self._variance = 1.

        # every input kernel to be one dimensional
        for k in self._kerns:
            assert k.input_dim==1, "input kernels must be one dimensional"

        # do some book keeping - figure out the _non-variance_ parameters,
        # and store their names.  The GPy convention seems to leave the 
        # scaling parameter named "_kernel_type_variance"
        self._variance_index = -np.ones(len(self._kerns), dtype=np.int)
        self._params_per_kernel = []
        for i, k in enumerate(self._kerns):
            gpy_param_names = k._get_param_names()
            kron_param_names = []
            for name in gpy_param_names: 
                if 'variance' not in name: 
                    kron_param_names.append("dim_%d_%s"%(i, name))
                else:
                    self._variance_index[i] = gpy_param_names.index(name)
            self._params_per_kernel.append(kron_param_names)

        print self._params_per_kernel

    def _gram_list(self, grids, params=None): 
        """ generate a list of gram matrices, one for each dimension, 
        given the fixed grid """
        assert len(grids) == len(self._kerns), "must input a grid for each dimension"

        if params is not None:
            self._set_params(params)

        scale, kern_hypers = self._slice_params(self._get_params())
        scale = np.power(scale, 1. / float(len(grids)))
        Ks = []
        for k, grid in zip(self._kerns, grids):
            Ks.append(scale * k.K(X = grid.reshape((-1, 1))))
        return Ks

    ########### Compatibility methods with GPy ###########################
    def _set_params(self, params):
        assert len(params) == len(self._get_param_names()), "num params input not equal to number required"
        scale, sub_params = self._slice_params(params)
        self._variance = scale
        for i, k in enumerate(self._kerns):
            if self._variance_index[i] != -1:
                mask = np.ones(len(k._get_params()), dtype=bool)
                mask[self._variance_index[i]] = 0
                params = np.ones(mask.shape)
                params[mask] = sub_params[i]
            else:
                params = sub_params[i]
            k._set_params(params)

    def _get_params(self):
        params = [[self._variance]]
        for i, k in enumerate(self._kerns):
            kr = k._get_params()
            if self._variance_index[i] != -1: 
                mask = np.ones(len(k._get_params()), dtype=bool)
                mask[self._variance_index[i]] = 0
                kr = kr[mask]
            params.append(kr)
        return np.hstack(params)

    def _slice_params(self, params): 
        """ separate out a flattened vector of hyperparameters to a 
        list of arrays of them (using kernel information) """
        kern_hypers = []
        scale = params[0]
        startI = 1
        for i, k in enumerate(self._kerns):
            endI = startI + len(self._params_per_kernel[i])
            kern_hypers.append( np.array(params[startI:endI]) )
            startI = endI
        return scale, kern_hypers

    def _get_param_names(self):
        param_names = ['kron_variance']
        kron_params = [p for kern_ps in self._params_per_kernel for p in kern_ps]
        param_names.extend(kron_params)
        return param_names
        ## this is a bit nasty: we want to distinguish between parts with the same name by appending a count
        #part_names = np.array([k.name for k in self._kerns], dtype=np.str)
        #counts = [np.sum(part_names == ni) for i, ni in enumerate(part_names)]
        #cum_counts = [np.sum(part_names[i:] == ni) for i, ni in enumerate(part_names)]
        #names = [name + '_' + str(cum_count) if count > 1 else name for name, count, cum_count in zip(part_names, counts, cum_counts)]
        #return sum([[name + '_' + n for n in k._get_param_names()] for name, k in zip(names, self._kerns)], [])

    def K(self, X, X2=None, which_parts='all'):
        """
        Compute the kernel function.

        :param X: the first set of inputs to the kernel
        :param X2: (optional) the second set of arguments to the kernel. If X2
                   is None, this is passed throgh to the 'part' object, which
                   handles this as X2 == X.
        :param which_parts: a list of booleans detailing whether to include
                            each of the part functions. By default, 'all'
                            indicates [True]*self.num_parts
        """
        raise Exception("""User should not ask for the full covariance of a 
                        kronecker matrix""")
        #if which_parts == 'all':
        #    which_parts = [True] * self.num_parts
        #assert X.shape[1] == self.input_dim
        #if X2 is None:
        #    target = np.zeros((X.shape[0], X.shape[0]))
        #    [p.K(X[:, i_s], None, target=target) for p, i_s, part_i_used in zip(self.parts, self.input_slices, which_parts) if part_i_used]
        #else:
        #    target = np.zeros((X.shape[0], X2.shape[0]))
        #    [p.K(X[:, i_s], X2[:, i_s], target=target) for p, i_s, part_i_used in zip(self.parts, self.input_slices, which_parts) if part_i_used]
        #return target

    def dK_dtheta(self, dL_dK, X, X2=None):
        pass


    #def hyper_prior_lnpdf(self, h): 
    #  """ prior over hyper parameters for this kernel - 
    #  using scale invariant for now """
    #  scale, hparams = self._chunk_hypers(h)
    #  lls = 0
    #  for d in range(len(hparams)):
    #    lls += self._kerns[d].hyper_prior_lnpdf(hparams[d])
    #  lls += gamma(2, scale=2).logpdf(scale)
    #  return lls

    def gen_prior(self, grids, params=None, nu=None):
        """ Efficiently draws a Multivariate Gaussian Random variable with 
        this covariance function (with parameters params). 
          Input: 
              grids  : grid locations for the covariance function
              params : covariance hyperparameters (optional)
              nu     : input randomness, marginally normal 0,1 RV's (optional)
        """
        # covariance mats for each dimension
        Ks = self._gram_list(grids=grids, params=params)

        # cache cholesky for each covariance
        Ls = [np.linalg.cholesky(K) for K in Ks]

        #generate spatial component and bias 
        Nz = np.prod( [len(g) for g in grids] )
        if nu is None: 
          nu = np.random.randn(Nz)
        return kron_mat_vec_prod(Ls, nu) 

    #def whiten_process(self, f, hypers, grids): 
    #  Ks = self._gram_list(hypers, grids)
    #  Ls = [np.linalg.cholesky(K) for K in Ks]
    #  Ls_inverse = [np.linalg.inv(L) for L in Ls]
    #  nu = kron_mat_vec_prod(Ls_inverse, f)
    #  return nu


if __name__=="__main__":
    import GPy
    import matplotlib.pyplot as plt
    kern_1 = GPy.kern.Matern32(input_dim=1)
    kern_2 = GPy.kern.Matern32(input_dim=1)
    kron_kernel = KronKernel([kern_1, kern_2])
    params = np.array([1., 3., 3.])
    kron_kernel._set_params(params)

    # draw a sample and visualize
    x1 = np.linspace(0, 30, 50)
    x2 = np.linspace(3, 40, 50)
    grids = [x1, x2]
    grams = kron_kernel._gram_list(grids)

    samp = kron_kernel.gen_prior(grids = grids).reshape((len(x1), len(x2)))

    fig, axarr = plt.subplots(2, 3)
    axarr[0,0].set_title('%s * %s'%(kron_kernel._kerns[0].name, 
                                  kron_kernel._kerns[1].name))
    axarr[0,0].imshow(samp.T,
                    extent=(x1.min(), x1.max(), x2.min(), x2.max()))
    axarr[0,0].set_aspect('auto')
    axarr[0,1].plot(x1, samp[:, 20])
    axarr[0,1].set_title('%s, l = %2.2f'%(kron_kernel._kerns[0].name, 
                                          kron_kernel._get_params()[1]))
    axarr[0,2].plot(x2, samp[20, :])
    axarr[0,2].set_title('%s, l = %2.2f'%(kron_kernel._kerns[1].name, 
                                        kron_kernel._get_params()[2]))
    fig.suptitle('%d x %d grid GP sample'%(len(x1), len(x2)))
    #fig.tight_layout()
    plt.show()

    mat_kern = GPy.kern.Matern32(input_dim=2)
    mat_kern._set_params(params)
    x1 = np.linspace(0, 30, 50)
    x2 = np.linspace(0, 40, 50)
    xx, yy = np.meshgrid(x1, x2)
    K = mat_kern.K(np.column_stack((xx.ravel(), yy.ravel())))
    V = len(x1) * len(x2)
    samp = np.linalg.cholesky(K + 1e-6*np.eye(V)).dot(np.random.randn(V)).reshape(xx.shape)

    axarr[1,0].imshow(samp.T)
    axarr[1,0].set_aspect('auto')
    axarr[1,1].plot(x1, samp[:,20])
    axarr[1,2].plot(x2, samp[20,:])
    plt.show()

    ####################################################
    # test to make sure generated processes are correct
    ####################################################
    x1 = np.linspace(0, 10, 20)
    x2 = np.linspace(3, 12, 20)
    xx, yy = np.meshgrid(x1, x2)
    V = len(xx.ravel())

    # typical covariance generation
    k_rbf = GPy.kern.rbf(input_dim=2)
    k_rbf['rbf_lengthscale'] = 1.
    k_rbf['rbf_variance'] = 1.
    K_slow = k_rbf.K(np.column_stack((xx.ravel(), yy.ravel())))

    # atypical covariance generation
    kron_rbf = KronKernel([GPy.kern.rbf(input_dim=1), GPy.kern.rbf(input_dim=1)])
    kron_rbf._set_params(np.array([1., 1., 1.]))
    grams = kron_rbf._gram_list([x2, x1])
    K_fast = kron_multi(grams)

    print "Gram matrix close?: ", np.allclose(K_slow, K_fast, atol=1e-5)
    nu = np.random.randn(V)
    K = k_rbf.K(np.column_stack((xx.ravel(), yy.ravel())))
    samp = np.linalg.cholesky(K).dot(nu).reshape(xx.shape)
    samp2 = kron_rbf.gen_prior(grids=[x2, x1], nu=nu).reshape(xx.shape)
    print np.allclose(samp, samp2, atol=1e-4)

    fig, axarr = plt.subplots(1,3)
    axarr[0].imshow(samp.T, interpolation='none')
    axarr[1].imshow(samp2.T, interpolation='none')
    a = axarr[2].imshow(samp.T - samp2.T, interpolation='none')
    plt.colorbar(a)
    plt.show()


