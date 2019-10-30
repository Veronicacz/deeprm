import numpy as np


class OU(object):

	def ouf(self, x, mu, theta, sigma, rand):
		noise = theta * (mu - x) + sigma * rand.randn(1)
		return noise