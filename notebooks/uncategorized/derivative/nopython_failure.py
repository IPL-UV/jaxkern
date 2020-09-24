import numba


@jit(nopython=True)
def rbf_full_numba(x_train, x_function, K, weights, gamma):
    
    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape
    
    derivative = np.zeros(shape=(n_test, n_train, d_dims))
    
    constant = - 2* gamma
    for itest in range(n_test):
        for itrain in range(n_train):
            w = weights[itrain]
            k = K[itest, itrain]
            for idim in range(d_dims):
                derivative[itest, itrain, idim] = \
                    w \
                    * (x_function[itest, idim] - x_train[itrain, idim]) \
                    * k
    
    derivative *= constant
    
    return derivative

if 