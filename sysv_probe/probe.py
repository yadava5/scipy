"""Call scipy-openblas32's dgesv/dsysv/dgetrf/dsytrf directly on the lhs from gh-26475.

Prints the OpenBLAS config, info from each routine, the pivots that the
routines test against zero, and how info changes under symmetric
permutations of the same (exactly singular) matrix.
"""
import ctypes
import glob
import itertools
import os
import platform
import sys

import numpy as np
import scipy_openblas32

pkgdir = os.path.dirname(scipy_openblas32.__file__)
cands = [p for p in glob.glob(os.path.join(pkgdir, "**", "*"), recursive=True)
         if "openblas" in os.path.basename(p) and p.endswith((".so", ".dylib", ".dll"))]
print("lib:", cands)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(cands[0]))
lib = ctypes.CDLL(cands[0])
lib.scipy_openblas_get_config.restype = ctypes.c_char_p
lib.scipy_openblas_get_corename.restype = ctypes.c_char_p

LHS = np.array([[ 0.        , -0.1732868 ,  2.77258872,  0.        ,  2.77258872,  4.02359478,  1.        , -1.        ,  1.        ],
       [-0.1732868 ,  0.        ,  0.91229649, -0.1732868 ,  0.91229649,  1.91531437,  1.        , -1.        ,  0.5       ],
       [ 2.77258872,  0.91229649,  0.        ,  0.        ,  0.        ,  0.        ,  1.        , -1.        , -1.        ],
       [ 0.        , -0.1732868 ,  0.        ,  0.        ,  0.        ,  0.69314718,  1.        , -1.        ,  0.        ],
       [ 2.77258872,  0.91229649,  0.        ,  0.        ,  0.        ,  0.        ,  1.        , -1.        , -1.        ],
       [ 4.02359478,  1.91531437,  0.        ,  0.69314718,  0.        ,  0.        ,  1.        ,  1.        , -1.        ],
       [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  0.        ,  0.        ,  0.        ],
       [-1.        , -1.        , -1.        , -1.        , -1.        ,  1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.5       , -1.        ,  0.        , -1.        , -1.        ,  0.        ,  0.        ,  0.        ]])
RHS = np.zeros((9, 1)); RHS[5] = 1.0
N = 9
i32 = ctypes.c_int32
P = ctypes.POINTER


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def gesv(a, b):
    a = np.asfortranarray(a.copy()); b = np.asfortranarray(b.copy())
    ipiv = np.zeros(N, np.int32); info = i32(0)
    lib.scipy_dgesv_(ctypes.byref(i32(N)), ctypes.byref(i32(1)), ptr(a), ctypes.byref(i32(N)),
                     ptr(ipiv), ptr(b), ctypes.byref(i32(N)), ctypes.byref(info))
    return info.value, b


def getrf(a):
    a = np.asfortranarray(a.copy())
    ipiv = np.zeros(N, np.int32); info = i32(0)
    lib.scipy_dgetrf_(ctypes.byref(i32(N)), ctypes.byref(i32(N)), ptr(a), ctypes.byref(i32(N)),
                      ptr(ipiv), ctypes.byref(info))
    return info.value, np.diag(a).copy()


def sysv(a, b, uplo=b"U"):
    a = np.asfortranarray(a.copy()); b = np.asfortranarray(b.copy())
    ipiv = np.zeros(N, np.int32); info = i32(0)
    work = np.zeros(1000); lwork = i32(1000)
    lib.scipy_dsysv_(ctypes.c_char_p(uplo), ctypes.byref(i32(N)), ctypes.byref(i32(1)), ptr(a),
                     ctypes.byref(i32(N)), ptr(ipiv), ptr(b), ctypes.byref(i32(N)), ptr(work),
                     ctypes.byref(lwork), ctypes.byref(info), ctypes.c_size_t(1))
    return info.value, b


def sytrf(a, uplo=b"U"):
    a = np.asfortranarray(a.copy())
    ipiv = np.zeros(N, np.int32); info = i32(0)
    work = np.zeros(1000); lwork = i32(1000)
    lib.scipy_dsytrf_(ctypes.c_char_p(uplo), ctypes.byref(i32(N)), ptr(a), ctypes.byref(i32(N)),
                      ptr(ipiv), ptr(work), ctypes.byref(lwork), ctypes.byref(info),
                      ctypes.c_size_t(1))
    # Smallest |D(k,k)| over the 1x1 pivot blocks (dsytrf sets info only when one is exactly 0).
    d = []
    k = 0 if uplo == b"L" else N - 1
    if uplo == b"L":
        while k < N:
            if ipiv[k] > 0:
                d.append(abs(a[k, k])); k += 1
            else:
                k += 2
    else:
        while k >= 0:
            if ipiv[k] > 0:
                d.append(abs(a[k, k])); k -= 1
            else:
                k -= 2
    return info.value, min(d)


print("platform:", platform.platform(), platform.machine(), "python", sys.version.split()[0])
print("config:  ", lib.scipy_openblas_get_config().decode())
print("corename:", lib.scipy_openblas_get_corename().decode())
print("rows 2 and 4 identical:", bool(np.array_equal(LHS[2], LHS[4])), "| symmetric:", bool(np.array_equal(LHS, LHS.T)))
print("numpy smallest singular value:", f"{np.linalg.svd(LHS, compute_uv=False)[-1]:.2e}")

ig, xg = gesv(LHS, RHS)
iu, xs = sysv(LHS, RHS)
il, xl = sysv(LHS, RHS, b"L")
print(f"info gesv = {ig}, sysv(U) = {iu}, sysv(L) = {il}")
info, u = getrf(LHS)
print(f"getrf info = {info}, U(info,info) = {u[info - 1]!r}" if info else
      f"getrf info = 0, min |U(i,i)| = {np.abs(u).min():.2e}")
for uplo in (b"U", b"L"):
    info, dmin = sytrf(LHS, uplo)
    print(f"sytrf({uplo.decode()}) info = {info}, smallest 1x1 pivot |D(k,k)| = {dmin:.2e}")
if iu == 0:
    print(f"sysv(U) solution: max|x| = {np.abs(xs).max():.2e}, "
          f"residual max|A x - b| = {np.abs(LHS @ xs - RHS).max():.2e}")

# Same matrix, rows and columns reordered: still exactly singular.
rng = np.random.default_rng(0)
perms = [rng.permutation(N) for _ in range(500)]
counts = {"gesv": 0, "sysv": 0}
for p in perms:
    A = LHS[np.ix_(p, p)]; b = RHS[p]
    counts["gesv"] += gesv(A, b)[0] > 0
    counts["sysv"] += sysv(A, b)[0] > 0
print(f"500 symmetric reorderings: gesv info>0 in {counts['gesv']}, sysv info>0 in {counts['sysv']}")
