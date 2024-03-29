import numpy as np
import itertools
import pyflann

import matplotlib.pyplot as plt
from util.data_process import plot_3d_points


"""
    This class represents a n-dimensional unit cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.

"""


class Space:

    def __init__(self, low, high, points):

        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        # print("dim:"+str(self._dimensions) ) #5
        self.__space = init_uniform_space([0.] * self._dimensions,
                                          [1.] * self._dimensions,
                                          points)
        self._flann = pyflann.FLANN() # https://blog.csdn.net/jcq521045349/article/details/78898971
        # FLANN: Fast Library for Approximate Nearest Neighbors
        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self.__space, algorithm='kdtree')
        # The method returns a dictionary containing the parameters used to construct the index.

    def search_point(self, point, k):
        p_in = self.import_point(point) 
        search_res, _ = self._flann.nn_index(p_in, k)  # def nn_index(self, qpts, num_neighbors = 1, **kwargs)
        # This method searches for the num_neighbors nearest neighbors of each point in testset using the index computed by 
        # build index. Additionally, a parameter called checks, denoting the number of times the index tree(s) should be recursivelly
        # searched, must be given.

        knns = self.__space[search_res]
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]
        return np.array(p_out)

    def import_point(self, point):
        return (point - self._low) / self._range

    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]

    def plot_space(self, additional_points=None):

        dims = self._dimensions

        if dims > 3:
            print(
                'Cannot plot a {}-dimensional space. Max 3 dimensions'.format(dims))
            return

        space = self.get_space()
        if additional_points is not None:
            for i in additional_points:
                space = np.append(space, additional_points, axis=0)

        if dims == 1:
            for x in space:
                plt.plot([x], [0], 'o')

            plt.show()
        elif dims == 2:
            for x, y in space:
                plt.plot([x], [y], 'o')

            plt.show()
        else:
            plot_3d_points(space)


class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        0, 1, 2, ..., n-2, n-1
    """

    def __init__(self, n):  # n: the number of the discrete actions
        super().__init__([0], [n - 1], n)

    def export_point(self, point):
        return super().export_point(point).astype(int)


def init_uniform_space(low, high, points):
    dims = len(low)
    # points_in_each_axis = round(points**(1 / dims)) # max actions^len

    # if continous action space more than one dimensional:
    # points_in_each_axis = np.float_power(points, 1./dims) 
    points_in_each_axis = np.float_power(points, 1./dims) 

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))
        # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
        # Return evenly spaced numbers over a specified interval.

        # Returns num evenly spaced samples, calculated over the interval [start, stop].

        # The endpoint of the interval can optionally be excluded.

    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    return np.array(space)
