#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from scipy.optimize import root_scalar
import sys

def thetabx(x, b):
    return 1.0 / (x + b)

def mat1d(mtype, s, h, ix, x, b, aphib, xlen, blen, Lx):
    gam = b[blen-1]  
    Gam = b[0]       
    aa = lil_matrix((xlen, xlen), dtype=float)
    
    for k in range(xlen):  
        for l in range(blen):
            ixkl = ix[k, l] - 1 
            v0 = Lx[k, l, 0]    
            v1 = Lx[k, l, 1]  
            kk0 = ixkl
            kk1 = ixkl + 1
            
            if kk0 >= xlen or kk1 >= xlen:
                print(f"Index out of bounds: kk0={kk0}, kk1={kk1}, xlen={xlen}")
                continue
            
            if mtype == -1:
                err = (s * (2 * s + 1) / gam**2) * np.exp(2 * s * h / gam) * v0 * v1 * h * h
            elif mtype == 0:
                err = 0.0
            elif mtype == 1:
                err = (s * (2 * s + 1) / (2 / gam + Gam)**2) * np.exp(-2 * s * h / gam) * v0 * v1 * h * h
            else:
                raise ValueError("mtype not -1,0,1")
            
            aphibpskl = aphib[k, l] ** (2 * s)
            aa[k, kk0] += aphibpskl * v0 * (1.0 - err)
            aa[k, kk1] += aphibpskl * v1 * (1.0 - err)
    
    return aa.tocsc()  # for eigs

def main():
    np.set_printoptions(precision=16)
    
    N = int(input("Input N, the number of subintervals: "))
    R = int(input("Input R, the number of coefficients in the continued fraction: "))
    blen = R
    b = np.zeros(blen, dtype=float)
    
    for j in range(blen):
        val = float(input("Input the values of the vector b in increasing order one at a time after each prompt: "))
        b[blen - j - 1] = val  # assign from largest to smallest index
    
    print("b:", b)
    c = 0.0
    d = 1.0 / b[blen-1]
    print("c:", c)
    print("d:", d)
    
    h = (d - c) / N
    xlen = N + 1
    x = np.linspace(c, d, xlen)
    
    #This part we compute all the combinations of the map 1/(n +x) for all n and all nodes x
    aphib = np.zeros((xlen, blen))
    for k in range(xlen):
        for l in range(blen):
            aphib[k, l] = 1.0 / np.abs(x[k] + b[l])
    
    ix = np.zeros((xlen, blen), dtype=int)
    Lx = np.zeros((xlen, blen, 2))
    
    for k in range(xlen):
        for l in range(blen):
            thetabxkl = thetabx(x[k], b[l])
            idx = int(np.floor((thetabxkl - c) / h)) + 1  # 1-based
            if idx < 1:
                print(f"ix(k,l) < 1: k={k+1}, l={l+1}, ix={idx}, thetabxkl={thetabxkl}")
            if idx > xlen:
                print(f"ix(k,l) > xlen: k={k+1}, l={l+1}, ix={idx}, thetabxkl={thetabxkl}")
            if idx == xlen:
                idx -= 1
            ix[k, l] = idx
            
            ixkl = ix[k, l] - 1  # 0-based for x
            thetabxkl = thetabx(x[k], b[l])
            Lx[k, l, 1] = (thetabxkl - x[ixkl]) / h  # right weight Lx[...,2] in MATLAB
            Lx[k, l, 0] = 1.0 - Lx[k, l, 1]          # left weight
            
    # Approximate
    def funa(s):
        A = mat1d(0, s, h, ix, x, b, aphib, xlen, blen, Lx)
        ev = eigs(A, k=1, which='LM', return_eigenvectors=False)[0].real
        return np.log(ev)
    
    sa = root_scalar(funa, bracket=[0.01, 0.99], xtol=1e-10).root
    A = mat1d(0, sa, h, ix, x, b, aphib, xlen, blen, Lx)
    muaa = eigs(A, k=1, which='LM', return_eigenvectors=False)[0].real
    muaa1 = muaa - 1
    print("Approximation of Hausdorff dimension s.")
    print(sa)
    print("The eigenvalue of the approximate matrix minus one.")
    print(muaa1)
    
    # Lower
    def funl(s):
        A = mat1d(-1, s, h, ix, x, b, aphib, xlen, blen, Lx)
        ev = eigs(A, k=1, which='LM', return_eigenvectors=False)[0].real
        return np.log(ev)
    
    sl = root_scalar(funl, bracket=[0.01, sa], xtol=1e-10).root
    As = mat1d(-1, sl, h, ix, x, b, aphib, xlen, blen, Lx)
    val, vec = eigs(As, k=1, which='LM')
    mual = val[0].real
    val = vec[:,0].real  # eigenvector
    mual1 = mual - 1
    
    if np.max(val) <= 0:
        val = -val
    
    minval = np.min(As @ val - val)
    
    eps = np.finfo(float).eps
    while mual1 <= 0 or minval < 0:
        sl -= eps
        As = mat1d(-1, sl, h, ix, x, b, aphib, xlen, blen, Lx)
        val, vec = eigs(As, k=1, which='LM')
        mual = val[0].real
        val = vec[:,0].real
        mual1 = mual - 1
        if np.max(val) <= 0:
            val = -val
        minval = np.min(As @ val - val)
    
    print("Lower bound on the Hausdorff dimension s.")
    print(sl)
    print("The eigenvalue of the lower matrix minus one.")
    print(mual1)
    print(minval)  # should be >=0
    
    # Upper
    def funu(s):
        A = mat1d(1, s, h, ix, x, b, aphib, xlen, blen, Lx)
        ev = eigs(A, k=1, which='LM', return_eigenvectors=False)[0].real
        return np.log(ev)
    
    su = root_scalar(funu, bracket=[sa, 0.99], xtol=1e-10).root
    Bs = mat1d(1, su, h, ix, x, b, aphib, xlen, blen, Lx)
    vau, vecu = eigs(Bs, k=1, which='LM')
    muau = vau[0].real
    vau = vecu[:,0].real  # eigenvector
    muau1 = muau - 1
    
    if np.max(vau) <= 0:
        vau = -vau
    
    minvau = np.min(vau - Bs @ vau)
    
    while muau1 >= 0 or minvau < 0:
        su += eps
        Bs = mat1d(1, su, h, ix, x, b, aphib, xlen, blen, Lx)
        vau, vecu = eigs(Bs, k=1, which='LM')
        muau = vau[0].real
        vau = vecu[:,0].real
        muau1 = muau - 1
        if np.max(vau) <= 0:
            vau = -vau
        minvau = np.min(vau - Bs @ vau)
    
    print("Upper bound on the Hausdorff dimension s.")
    print(su)
    print("The eigenvalue of the upper matrix minus one.")
    print(muau1)
    print(minvau)  # should be >=0

if __name__ == "__main__":
    main()


# In[ ]:




