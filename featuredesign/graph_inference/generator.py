#random graph generation funciton
# return adjacency matrix
#generate BA graph, ER graph.
from pygsp import *
import numpy as np
from scipy.linalg import expm
#expm is matrix exponential function

def graph_generator(N,p,m0=0,m=0,type='ER'):
    if type == 'ER':
        G = graphs.ErdosRenyi(N=N, p=p, directed=0, self_loops=0)
    elif type == 'BA':
        G = graphs.BarabasiAlbert(N=N, m0=m0, m=m)
    elif type == 'SBM':
        G = graphs.StochasticBlockModel(N = N,k = 2, p = 0.7, q = 0.1)
    return G.W.toarray()
    # G.W is the weighted matrix represented by array matrix


def signal_generator(W,x,n,f):
    # W is adjacency matrix, x is input signal(column vector), f is graph filter
    # consider 2 graph filters: exp(-W) and I + W + W^2

    if f == 'lowpass_exp_filter':
        m = expm(0.5*W)
        return np.dot(m, x), m.dot(m)
    elif f == 'quadratic_filter':
        idm = np.eye(n,n)
        np.fill_diagonal(idm, 1)
        m = idm+W+W.dot(W)
        return np.dot(m, x), m.dot(m)
    elif f == 'highpass_exp_filter':
        m = expm(-W)
        return np.dot(m, x), m.dot(m)
