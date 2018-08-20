#encoding=utf-8
from six.moves import reduce

from collections import namedtuple
import numpy as np
import scipy.sparse as sp

from mmath import xyt_inv_mult, xyt_mult, MultiVariateGaussian

import pyximport; pyximport.install()
from perfx import XYTConstraint_residual, XYTConstraint_jacobians
from perfx import normalize_angle_range
import math



#
# Some specializations for performance
#

def _hstack2d(arr_collection):
    """ faster than `np.hstack` because it avoids calling `np.atleast1d` """
    return np.concatenate(arr_collection, axis=1)


# hstack: 水平(按列顺序)把数组给堆叠起来.vstack: 垂直...
def _hstack1d(arr_collection):
    """ faster than `np.hstack` because it avoids calling `np.atleast1d` """
    return np.concatenate(arr_collection, axis=0)


# 这种表示coo_distinct_matrix继承自sp.coo_matrix.
class coo_distinct_matrix(sp.coo_matrix):
    """ A COO sparse matrix that assumes that there are no duplicate entries """

    def sum_duplicates(self):
        pass


#--------------------------------------
class VertexXYT(object):
#--------------------------------------
    def __init__(self, xyt0):
        self.state0 = xyt0


#--------------------------------------
class _MatrixDataIJV(object):
#--------------------------------------
    """ Encapsulation for matrix (i,j,v) data stored in a contiguous
    record array.
    """
    # __slots__限制了类的访问属性，只能有_data访问.
    __slots__ = [ '_data' ]

    def __init__(self, ijv_records):
        # 后面这一大串是代表这dtype
        # '>'为big-edian, '<'为little-endian. i为'int32'
        # 每一行表示为i, j, 以及数值, 类型为: int32, int32, float64(8字节).
        self._data = np.array(ijv_records, [ ('i', '<i4'), ('j', '<i4'), ('v', '<f8') ])

    # 注意classmethod和staticmethod的区别等.
    @classmethod
    def from_ijv(cls, i, j, v):
        records = [ (ii, jj, vv) for ii, jj, vv in zip(i, j, v) ]
        return cls(records)

    @classmethod
    def from_dense(cls, M, i_offset=0, j_offset=0):
        # np.indices表示的是x,y上面的位置构成的矩阵.  M.shape表示矩阵形状.
        # 后面是把每个数都加上i_offset和j_offset.
        i, j = np.indices(M.shape).reshape(2, -1) + np.c_[[i_offset, j_offset]]
        # ravel实现降维，把多维变成一维.
        return cls.from_ijv(i, j, M.ravel())

    def indices(self):
        return self._data.view(dtype=np.int32).reshape((-1, 4))[:,:2].T

    @property
    def i(self):
        return self.indices()[0]

    @property
    def j(self):
        return self.indices()[1]

    @property
    def v(self):
        return self._data.view(dtype=np.float64).reshape((-1, 2))[:,1].T

    @v.setter
    def v(self, value):
        self._data.view(dtype=np.float64).reshape((-1, 2))[:,1] = value

    def offset(self, i, j):
        x = self._data.copy()
        x.view(dtype=np.int32).reshape((-1, 4))[:,:2] += np.array([i, j], dtype=np.int32)
        return self.__class__(x)


def _stack_ijv(ijv_data_list):
    return _MatrixDataIJV(np.concatenate([ x._data for x in ijv_data_list ]))


#--------------------------------------
class XYTConstraint(object):
#--------------------------------------
    """
    Constrain the transformation between two `VertexXYT`s.

    `xyt` is the rigidbody transformation `T`, between the vertices
    `v_out` and `v_in`, expressed using its parameters of x:
    displacement, y: displacement and theta: rotation

    This constraint constrains the `xyt` between `v_out` and `v_in` to
    be distributed according to the specified gaussian distribution.
    """
    _DOF = 3

    def __init__(self, v_out, v_in, gaussian):
        self._vx = [ v_out, v_in ]
        # _gaussian为原始的信息矩阵的逆.
        self._gaussian = gaussian
        # 构建信息矩阵的稀疏矩阵?
        self._Sigma_ijv = _MatrixDataIJV.from_dense(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def residual(self):
        """
        Compute the difference in transformation implied by this edge
        from `self._gaussian.mu`.
        """
        stateA, stateB = self._vx[0].state, self._vx[1].state
        return XYTConstraint_residual(self._gaussian.mu, stateA, stateB)

    def chi2(self):
        z = self.residual()
        # print z
        return reduce(np.dot, [ z.T, np.eye(3,3), z ])
        # return reduce(np.dot, [ z.T, self._gaussian.P, z ])

    def uncertainty(self, roff=0, coff=0):
        return self._Sigma_ijv.offset(roff, coff)

    def jacobian(self, roff=0):
        """
        Compute the jacobian matrix of the residual error function
        evaluated at the current states of the connected vertices.

        returns the sparse Jacobian matrix entries in triplet format
        (i,j,v). The row index of the entries is offset by `roff`.

        It is useful to specify `roff` when this Jacobian matrix is
        computed as a sub-matrix of the graph Jacobian.
        """
        Ja_, Jb_ = XYTConstraint_jacobians(self._vx[0].state, self._vx[1].state)
        #cdef double mean_matrix_inv[9];
        # mean_matrix[:] = [cos(mean[2]), -sin(mean[2]), mean[0],
        #                   sin(mean[2]),  cos(mean[2]), mean[1],
        #                   0,             0,            1.]
        mean = self._gaussian.mu
        mean_matrix_inv = np.array([(math.cos(mean[2])), (math.sin(mean[2])), 0,
                                    (-math.sin(mean[2])), (math.cos(mean[2])), 0,
                                    0                  , 0                 , 1])
        Ja = np.dot(mean_matrix_inv.reshape(3,3) , np.array(Ja_).reshape(3,3))
        Jb = np.dot(mean_matrix_inv.reshape(3,3) , np.array(Jb_).reshape(3,3))
        #print Ja, Jb
        #print Ja_
        #print Jb_
        #print mean_matrix_inv
        #print Ja

        if self._jacobian_ijv_cache is None:
            ndx0 = self._vx[0]._graph_state_ndx
            ndx1 = self._vx[1]._graph_state_ndx
            # 构建一个更大的稀疏矩阵.
            self._jacobian_ijv_cache = _stack_ijv([
                            _MatrixDataIJV.from_dense(Ja, roff, ndx0),
                            _MatrixDataIJV.from_dense(Jb, roff, ndx1)
                        ])

        # print np.array(self._jacobian_ijv_cache.v)
        # print np.array(Ja+Jb)
        self._jacobian_ijv_cache.v = list(Ja.reshape(1,-1)[0]) + list(Jb.reshape(1,-1)[0])
        return self._jacobian_ijv_cache


#--------------------------------------
class AnchorConstraint(object):
#--------------------------------------
    """
    Anchors the `xyt` parameters of a vertex `v` to conform to a
    gaussian distribution. The most common use of this edge type is to
    anchor the `xyt` of the first node in a SLAM graph to a fixed value.
    This prevents the graph solution from drifting arbitrarily.
    """
    _DOF = 3

    def __init__(self, v, gaussian):
        self._vx = [v]
        self._gaussian = gaussian
        self._Sigma_ijv = _MatrixDataIJV.from_dense(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def residual(self, aggregate_state=None):
        r = self._gaussian.mu - self._vx[0].state
        r[2] = normalize_angle_range(r[2])
        return r

    def chi2(self):
        return self._gaussian.chi2(self._vx[0].state)

    def uncertainty(self, roff=0, coff=0):
        return self._Sigma_ijv.offset(roff, coff)

    def jacobian(self, roff=0, eps=1e-5):
        """
        Compute the jacobian matrix of the residual error function
        evaluated at the current states of the connected vertices.

        returns the sparse Jacobian matrix entries in triplet format
        (i,j,v). The row index of the entries is offset by `roff`.

        It is useful to specify `roff` when this Jacobian matrix is
        computed as a sub-matrix of the graph Jacobian.
        """
        if self._jacobian_ijv_cache is None:
            J = -np.eye(3)
            ndx = self._vx[0]._graph_state_ndx
            self._jacobian_ijv_cache = _MatrixDataIJV.from_dense(J, roff, ndx)

        return self._jacobian_ijv_cache


GraphStats = namedtuple('GraphStats', ['chi2', 'chi2_N', 'DOF'])


#--------------------------------------
class Graph(object):
#--------------------------------------
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        # 把vertices里面的数据连接起来.
        self.state = np.concatenate([ v.state0 for v in self.vertices ])

        # The vertices get views into the graph's state
        # 记录了每一组数据的末端标记. [3, 6, 9, ..., 3*n]
        slice_end = np.cumsum([ len(v.state0) for v in self.vertices ]).tolist()
        # 同上，起点. [0, 3, 6, 9, ..., 3*(n-1)]
        slice_start = [0] + slice_end[:-1]
        for v, i, j in zip(self.vertices, slice_start, slice_end):
            v.state = self.state[i: j]
            v._graph_state_ndx = i

    def anchor_first_vertex(self):
        if hasattr(self, '_anchor') == False:
            v0 = self.vertices[0]

            mu = v0.state.copy()
            P = 1000. * np.eye(len(mu))
            self._anchor = AnchorConstraint(v0, MultiVariateGaussian(mu, P))

            self.edges.append(self._anchor)

    def get_stats(self):
        # 选定所有非锚点的边.
        original_edges = [ e for e in self.edges if e is not self._anchor ]

        # 5453*3 - 3500*3 = 5859, 这个DOF到底怎么理解?
        DOF = sum(e._DOF for e in original_edges) - len(self.state)
        chi2 = sum(e.chi2() for e in original_edges)

        return GraphStats(chi2, chi2/DOF, DOF)

    def get_linearization(self):
        """
        Linearizes the non-linear constraints in this graph, at its
        current state `self.state`, to produce an approximating linear
        system.

        Returns:
        --------
            `W`:
                Weighting matrix for the linear constraints
            `J`:
                Jacobian of the system at `self.state`.
            `r`:
                vector of stacked residuals

        The linear system `W J x = W r` captures an approximate linear
        model of the graph constraints that is valid near the current
        state of the graph.
        """
        edges = self.edges
        residuals = [ e.residual() for e in edges ]

        # For each edge jacobian, compute the its row index in the
        # graph jacobian `J`
        residual_lengths = [ len(r) for r in residuals ]
        row_offsets = [0,] + np.cumsum(residual_lengths).tolist()

        # Stack edge jacobians to produce system jacobian
        j_coo = _stack_ijv([ e.jacobian(roff=r) for r, e in zip(row_offsets, edges) ])
        J = coo_distinct_matrix((j_coo.v, (j_coo.i, j_coo.j)))
        # np.savetxt("a.txt", J.toarray(), fmt="%.6e")

        # Stack edge weights to produce system weights
        w_coo = _stack_ijv([ e.uncertainty(r, r) for r, e in zip(row_offsets, edges) ])
        W = coo_distinct_matrix((w_coo.v, (w_coo.i, w_coo.j))).tocsc()

        r = _hstack1d(residuals)

        return W, J, r
