#%%
import numpy as np
from pygsp import *
import matplotlib.pyplot as plt

def agtlgrfunc(lambda1, lambda2, lambda3, s, y, z, q, alpha, rho, cn):
    m = np.size(s, 1)

    return np.sum(s) - alpha * np.sum(np.log(q)) + np.sum(lambda1 * (s - y)) + np.sum(
        lambda2 * (cn.dot(s) - s.dot(cn) - z)) \
           + np.sum(lambda3 * (q - np.sum(s, 1).reshape((m, 1)))) \
           + rho / 2 * np.sum(np.square(s - y)) \
           + rho / 2 * np.sum(np.square(cn.dot(s) - s.dot(cn) - z)) \
           + rho / 2 * np.sum(np.square(q - np.sum(s, 1).reshape((m, 1))))


def funvalue(s, alpha, beta):
    return beta / 2 * np.linalg.norm(s, 'fro') ** 2 + np.sum(s) - alpha * np.sum(np.log(np.sum(s, 1)))


def projection2s(sp):
    sp = 0.5 * (sp + sp.T)
    np.fill_diagonal(sp, 0)
    sp[sp < 0] = 0
    return sp


def projection2s1(sp):
    sp = 0.5 * (sp + sp.T)
    np.fill_diagonal(sp, 0)
    sp[sp < 0] = 0
    temp = list(sp[:, 0])
    temp = temp[1:len(temp)]
    s = sorted(temp, reverse=True)
    tmpsum = 0
    m = np.shape(sp)[0] - 1
    bget = False
    temp = np.array(temp)
    s = np.array(s)
    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break
    if ~bget:
        tmax = (tmpsum + s[m - 1] - 1) / m
    temp = temp - tmax
    temp[temp < 0] = 0
    sp[0, 1:] = temp
    sp[1:, 0] = temp
    return sp


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        return v
        # best projection: itself!
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = float(cssv[rho] - s) / (rho + 1)
    w = (v - theta).clip(min=0)
    return w


def projection2l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    m1 = v.shape[0]
    m2 = v.shape[1]
    v = v.reshape(m1 * m2)
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        w = v.reshape((m1, m2))
        return w
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    w = w.reshape((m1, m2))
    return w


def lADMM(Hn, s0, Z0, q0, lambda20, lambda30, alpha, delta, rho, tau1, epsilon=1e-6, kMax=5000):
    # Hn: the estimated covariance matrix; alpha,delta: parameters in the model
    # s0, Z0, q0, lambda20, lambda30: the initial points. A candidate set of values is
    # s0 = np.ones((m,m))
    # q0 = s0.dot(np.ones((m,1)))
    # Z0 = s0
    # lambda20 = np.zeros((m,m))
    # lambda30 = np.zeros((m,1))
    # rho: initial rho in the algorithm, will be updated adaptively in the iterations
    # tau1: a hyper-parameter in the algorithm, ranging from 0.5-1.  a rule-of-thumb value is 0.7. can be set larger if the algorithm doesn't converge.

    H2 = Hn.dot(Hn)
    m = np.size(Hn, 1)

    # tau = tau1*pow(np.linalg.norm(H,ord = 2),2)
    tau = tau1 * (pow(np.linalg.norm(Hn, ord=2), 2) + m)

    # funcvalues = np.zeros(kMax)
    r = np.zeros(kMax)
    ss = np.zeros(kMax)
    rrho = np.zeros(kMax)

    # update parameter for rho
    threh = 5
    upmul = 2
    dowmul = 2

    for i in range(kMax):
        rrho[i] = rho

        C = np.ones((m, m)) + Hn.dot(lambda20) - lambda20.dot(Hn) - lambda30.dot(np.ones((1, m))) \
            + rho * (s0.dot(H2) + H2.dot(s0) - 2 * Hn.dot(s0).dot(Hn) + Z0.dot(Hn) - Hn.dot(
            Z0) + s0.dot(np.ones((m, m))) - q0.dot(np.ones((1, m))))
        s = s0 - C / (rho * tau)
        s = projection2s(s)

        Z = (Hn.dot(s) - s.dot(Hn)) + 1 / rho * lambda20
        Z = min(delta / np.linalg.norm(Z, 'fro'), 1) * Z

        v = np.sum(s, 1).reshape((m, 1)) - (lambda30) / rho
        q = 0.5 * (v + np.sqrt(np.square(v) + 4 * alpha / rho))

        lambda20 = lambda20 + rho * (Hn.dot(s) - s.dot(Hn) - Z)
        lambda30 = lambda30 + rho * (q - s.dot(np.ones((m, 1))))

        # calculate residual
        r[i] = np.sqrt(pow(np.linalg.norm(Hn.dot(s) - s.dot(Hn) - Z, 'fro'), 2) +
                       pow(np.linalg.norm(q - s.dot(np.ones((m, 1)))), 2))
        ss[i] = np.linalg.norm(rho * Hn.dot(
            Z0 - Z) - rho * (Z0 - Z).dot(Hn) + rho * (q0 - q).dot(np.ones((1, m))), 'fro')

        # update
        s0 = s
        q0 = q
        Z0 = Z

        if (r[i] < m * epsilon) & (ss[i] < m * epsilon):
            break
        elif (r[i] > threh * ss[i]) & (i % 100 == 1):
            rho = np.min([upmul * rho, 1e5])
        elif (ss[i] > threh * r[i]) & (i % 100 == 1):
            rho = np.max([rho / dowmul, 1e-5])

    #        funcvalues[i] = funvalue(s0,alpha,0)

    if i == kMax - 1:
        print("may not converge")

    # funcvalues = funcvalues[0:i]
    rrho = rrho[0:i]
    ss = ss[0:i]
    r = r[0:i]
    # return s, q, Y, Z, ss,r,rrho,funcvalues
    return s, ss, r, rrho
# linearized ADMM to solve the original problem
    