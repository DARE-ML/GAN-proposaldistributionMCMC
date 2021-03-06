import numpy as np

np.random.seed(1234)
_2PI = 2. * np.pi

def simplex_coordinates( m ):
    # This function is adopted from the Simplex Coordinates library
    # https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.html
    x = np.zeros ( [ m, m + 1 ] )

    for j in range ( 0, m ):
        x[j,j] = 1.0

    a = ( 1.0 - np.sqrt ( float ( 1 + m ) ) ) / float ( m )

    for i in range ( 0, m ):
        x[i,m] = a
    c = np.zeros ( m )
    for i in range ( 0, m ):
        s = 0.0
        for j in range ( 0, m + 1 ):
            s = s + x[i,j]
        c[i] = s / float ( m + 1 )

    for j in range ( 0, m + 1 ):
        for i in range ( 0, m ):
            x[i,j] = x[i,j] - c[i]
    s = 0.0
    for i in range ( 0, m ):
        s = s + x[i,0] ** 2
        s = np.sqrt ( s )

    for j in range ( 0, m + 1 ):
        for i in range ( 0, m ):
            x[i,j] = x[i,j] / s
    return x


def var2cov(bot_dim, ngmm):
    cov = np.zeros((bot_dim, bot_dim))
    for k_ in range(bot_dim):
        cov[k_, k_] = 1.
    sigma_real_batch = []
    for c in range(ngmm):
        sigma_real_batch.append(cov)
    return np.array(sigma_real_batch, dtype=np.float32).squeeze().astype('float32') * .25

def simplex_params(bot_dim):
    ngmm = bot_dim + 1
    mu_real_batch = simplex_coordinates(bot_dim)
    sigma_real = var2cov( bot_dim, ngmm)
    mu_real = np.array(mu_real_batch.T, dtype=np.float32)
    w_real = (np.ones((ngmm,)) / ngmm).astype('float32')
    #print(mu_real_batch)
    #print(sigma_real)
    #print(mu_real)
    #print(w_real)
    return mu_real.astype('float32'), sigma_real.astype('float32'), w_real.astype('float32')

def get_noise(batchsize=500, zdim=2):
    return np.random.uniform(-1.0, 1.0, size=(batchsize, zdim)).astype(np.float32)

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    #triangle = np.array(
    #           [[ 0.8639503  ,-0.23149478],
    #            [-0.23149478 , 0.8639503 ],
    #            [-0.6324555  ,-0.6324555 ]]
    #)
    #plt.plot(triangle[:,0],triangle[:,1])
    #plt.show()
    #for i in range(3):
    #    print(sum(np.power(triangle[i]-triangle[(i+1)%3],2)))

    if True:
        for dim in range(1,5):
            print("------------ ",dim, " ------------")
            mu_sgmm, sigma_sgmm, w_sgmm = simplex_params(dim)
            print(mu_sgmm.shape,sigma_sgmm.shape,w_sgmm.shape)
            print(mu_sgmm)
            print(sigma_sgmm)
            print(w_sgmm)
            #print(mu_sgmm,sigma_sgmm,w_sgmm)
        