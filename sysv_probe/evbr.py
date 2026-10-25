import numpy as np
from scipy.linalg.lapack import get_lapack_funcs
import scipy, threadpoolctl
print("scipy", scipy.__version__)
lhs = np.array([[ 0.        , -0.1732868 ,  2.77258872,  0.        ,  2.77258872,  4.02359478,  1.        , -1.        ,  1.        ],
       [-0.1732868 ,  0.        ,  0.91229649, -0.1732868 ,  0.91229649,  1.91531437,  1.        , -1.        ,  0.5       ],
       [ 2.77258872,  0.91229649,  0.        ,  0.        ,  0.        ,  0.        ,  1.        , -1.        , -1.        ],
       [ 0.        , -0.1732868 ,  0.        ,  0.        ,  0.        ,  0.69314718,  1.        , -1.        ,  0.        ],
       [ 2.77258872,  0.91229649,  0.        ,  0.        ,  0.        ,  0.        ,  1.        , -1.        , -1.        ],
       [ 4.02359478,  1.91531437,  0.        ,  0.69314718,  0.        ,  0.        ,  1.        ,  1.        , -1.        ],
       [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  0.        ,  0.        ,  0.        ],
       [-1.        , -1.        , -1.        , -1.        , -1.        ,  1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.5       , -1.        ,  0.        , -1.        , -1.        ,  0.        ,  0.        ,  0.        ]])
rhs = np.zeros((9, 1)); rhs[5] = 1.0
dsysv = get_lapack_funcs('sysv', dtype=np.float64, ilp64="preferred")
dgesv = get_lapack_funcs('gesv', dtype=np.float64, ilp64="preferred")
r_ge = dgesv(lhs, rhs)
r_sy = dsysv(lhs, rhs)
for d in threadpoolctl.threadpool_info():
    print(d.get("internal_api"), d.get("version"), d.get("architecture"), d.get("filepath"))
print(lhs[2, :] - lhs[4, :])
print("info = ", r_ge[-1], r_sy[-1])
