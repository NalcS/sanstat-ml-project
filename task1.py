import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#D_lrn: ti = w0 + w1*xi + \epsilon = −1.2 + 0.9xi + \epsilon,  x = [−1, −0.99, . . . , 0.99, 1] 
#D_tst: ti = w0 + w1*xi + \epsilon = −1.2 + 0.9xi + \epsilon,  x = [−1.5, −1.4, . . . , −1.1, 1.1, . . . , 1.5] 
#\epsilon ∼ N (μ, σ2) , where μ = 0 and σ2 = 0.2


#1)


alpha = 2

covariance = (1 / alpha) * np.eye(2)

w_0_distribution = np.arange(-3, 3, 0.1)
w_1_distribution = np.arange(-3, 3, 0.1)

#create 2D grid
[W_0, W_1] = np.meshgrid(w_0_distribution, w_1_distribution)

positions = np.dstack((W_0, W_1))

thing_in_between = multivariate_normal(np.array([0,0]), covariance)

W_prior = thing_in_between.pdf(positions)

fig, ax = plt.subplots(1, 1)

#Z = np.cos(X / 2) + np.sin(Y / 4)
#multivariate_normal(mean, covariance)
#Z = multivariate_normal(W_0, W_1, 0, 1)

#plots contour
ax.contourf(W_0, W_1, W_prior)

ax.set_title('Contour Plot')
ax.set_xlabel('w_0')
ax.set_ylabel('w_1')

plt.show()