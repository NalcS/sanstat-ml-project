import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#D_lrn: ti = w0 + w1*xi + \epsilon = −1.2 + 0.9xi + \epsilon,  x = [−1, −0.99, . . . , 0.99, 1] 
#D_tst: ti = w0 + w1*xi + \epsilon = −1.2 + 0.9xi + \epsilon,  x = [−1.5, −1.4, . . . , −1.1, 1.1, . . . , 1.5] 
#\epsilon ∼ N (μ, σ2) , where μ = 0 and σ2 = 0.2


#1)

alpha = 2

covariance = (1 / alpha) * np.eye(2)
mu = np.array([0,0])

w_0_distribution = np.arange(-3, 3, 0.1)
w_1_distribution = np.arange(-3, 3, 0.1)

#create 2D grid
[W_0, W_1] = np.meshgrid(w_0_distribution, w_1_distribution)

positions = np.dstack((W_0, W_1))

W_prior = multivariate_normal(mu, covariance).pdf(positions)


#plot contour
plt.contourf(W_0, W_1, W_prior)
plt.show()