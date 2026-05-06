import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

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
#plt.contourf(W_0, W_1, W_prior)
#plt.title("Contour of prior")
#plt.show()






#2)

#True parameters (hidden from model)
w0_true = -1.2
w1_true = 0.9

#noise
sigma2 = 0.2
beta = 1 / sigma2

#generate data
x = np.linspace(-1, 1, 100)
t = w0_true + w1_true * x + np.random.normal(0, np.sqrt(sigma2), size=len(x))



#x_samples = x[sample_index]
#t_samples = t[sample_index]

#probabilities = np.ones((len(w_0_distribution), len(w_1_distribution)))

#for w0_i in range(len(w_0_distribution)):
#    for w1_i in range(len(w_1_distribution)):
#        for i in range(len(x_samples)):
#            w_x = w_0_distribution[w0_i] + w_1_distribution[w1_i] * x_samples[i]
#            probabilities[w1_i][w0_i] *= norm(w_x, np.sqrt(sigma2)).pdf(t_samples[i])


training_size = 3

sample_index = np.random.choice(len(x), training_size)
x_samples = x[sample_index]
t_samples = t[sample_index]

probabilities = np.ones_like(W_0)

for i in range(training_size):
    W_pos = W_0 + W_1 * x_samples[i]
    rv = norm(t_samples[i], np.sqrt(sigma2))
    probabilities *= rv.pdf(W_pos)


plt.contourf(W_0, W_1, probabilities)
plt.title("Contour of prior")
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.show()