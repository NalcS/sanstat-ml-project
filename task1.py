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


#plot contour for prior
plt.contourf(W_0, W_1, W_prior)
plt.title("Contour of prior")
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.show()






#2)

#True parameters (hidden from model)
w0_true = -1.2
w1_true = 0.9

#noise
sigma2 = 0.2
beta = 1 / sigma2

#generate data
x = np.linspace(-1, 1, 201)
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

sample_index = np.random.choice(len(x), training_size, replace=False)
x_samples = x[sample_index]
t_samples = t[sample_index]

probabilities = np.ones_like(W_0)

for i in range(training_size):
    W_pos = W_0 + W_1 * x_samples[i]
    rv = norm(t_samples[i], np.sqrt(sigma2))
    probabilities *= rv.pdf(W_pos)


#plot contour for likelyhood
plt.contourf(W_0, W_1, probabilities)
plt.title("Contour of likelihood")
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.show()



#3)
#we want to calculate the posterior based on the prior and the likelihood

x_ext = np.column_stack((np.ones(len(x_samples)), x_samples))

#first calculate the mean and covariance using #27 and #28
posterior_covariance = np.linalg.inv(alpha * np.eye(2) + beta * (x_ext.T @ x_ext))

posterior_mean = beta * ((posterior_covariance @ x_ext.T) @ t_samples)


W_posterior = multivariate_normal(posterior_mean, posterior_covariance).pdf(positions)

#plot contour for prior
plt.contourf(W_0, W_1, W_posterior)
plt.title("Contour of posterior")
plt.xlabel('w_0')
plt.ylabel('w_1')
plt.show()



#4)
posterior_sample_amount = 5
posterior_sample_index = multivariate_normal(posterior_mean, posterior_covariance).rvs(posterior_sample_amount)

x_test = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, 1.1, 1.2, 1.3, 1.4, 1.5])
t_test = w0_true + w1_true * x_test + np.random.normal(0, np.sqrt(sigma2), size=len(x_test))

x = np.linspace(-1.5, 1.5, 201)
for i in range(posterior_sample_amount):
    y = posterior_sample_index[i][0] + posterior_sample_index[i][1] * x
    plt.plot(x, y)
plt.plot(x_samples, t_samples, 'o', color='blue', label='Training')
plt.plot(x_test, t_test, 'o', color='red',label='Testing')

plt.plot(x, w0_true + w1_true * x, '--', color='black', label='The True Model') #the true model
plt.legend()
#plt.show()


#5)

x_test_ext = np.column_stack((np.ones(len(x_test)), x_test))

test_mu = posterior_mean.T @ x_test_ext.T

test_variance = (1/beta) + np.diag(((x_test_ext @ posterior_covariance) @ x_test_ext.T))

plt.errorbar(x_test, test_mu, yerr=np.sqrt(test_variance),
             fmt='s', color='black', capsize=5, capthick=1.5,
             label='Predictive mean +/- 1 std')
#plt.show()


#6)

w_ML = np.linalg.inv(x_ext.T @ x_ext) @ x_ext.T @ t_samples

print(w_ML)
print()
print(t_samples)
print()
print(w_ML)
print()
print(x_ext.T)

beta_ML = (len(t_samples)) / np.sum((t_samples - (w_ML @ x_ext.T))**2)
print(beta_ML)

y = w_ML[0] + w_ML[1] * x
plt.plot(x, y, '--', 'green')
plt.show()