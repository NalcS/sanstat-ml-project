import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

#1)

#ti = wTφ(xi) + \epsilon = w0 + w1x1i2 + w2x2i3 + \epsilon , where \epsilon ∼ N (0, σ2)

sigma2 = 0.3

w_true = np.array([0, 2.5, -0.5])

#creating the x matrix with the x values and the x_ext matrix with 1 also
#   ones, x1 repeated m times x2 rep m times... xn rep m times, y1...ym repeated n times
x_in = np.linspace(-1, 1, 41)

#create 2D grid
[X_1, X_2] = np.meshgrid(x_in, x_in)

#   tried to make it 1 matrix but was better with grid for the plotting, perhaps useful for later tasks though
#col2 = np.repeat(x_in, 41)
#col3 = np.tile(x_in, 41)
#col1 = np.ones(len(col3))
#x =  np.column_stack((col2, col3))
#x_ext = np.column_stack((col1, col2, col3))

#create t:
t_noise = np.random.normal(0, np.sqrt(sigma2), size=X_1.shape)
T = w_true[0] + (w_true[1] * (X_1**2)) + (w_true[2] * (X_2**3)) + t_noise

plt.contourf(X_1, X_2, T)
plt.show()



#2)

x1_flat = X_1.flatten()
x2_flat = X_2.flatten()
t_flat = T.flatten()

test_condition = (np.abs(x1_flat) > 0.3) | (np.abs(x2_flat) > 0.3) 
training_condition = ~test_condition

#print(test_condition)
#print(training_condition)

x1_test = x1_flat[test_condition]
x2_test = x2_flat[test_condition]
t_test = t_flat[test_condition]

#add extra noise to t_test

extra_noise = np.random.normal(0, 0.25*np.sqrt(sigma2), size=t_test.shape)

t_test = t_test + extra_noise

#training sample
x1_training = x1_flat[training_condition]
x2_training = x2_flat[training_condition]
t_training = t_flat[training_condition]

print(x1_test)
print()
print(x2_test)
print()
print(t_test)
#plot
plt.scatter(x1_test, x2_test, c=t_test, cmap='viridis', label='Test Data')
plt.show()

plt.scatter(x1_training, x2_training, c=t_training, cmap='viridis', label='Training Data')
plt.show()




#sampling
#   |x_1|>0.3 or |x_2|>0.3
#       for test data
#   rest is for training

#x_test_in = np.array(list(filter(lambda x: np.abs(x) > 0.3000001, x_in))) #has to check 0.3000001 because float bs

#[X_test_1, X_test_2] = np.meshgrid(x_test_in, x_test_in)

#t_test_noise = 1.25 * np.random.normal(0, np.sqrt(sigma2), size=X_test_1.shape) #extra noise
#T_test = w_true[0] + (w_true[1] * (X_test_1**2)) + (w_true[2] * (X_test_2**3)) + t_test_noise

#print(T_test)

#plt.contourf(X_test_1, X_test_2, T_test)
#plt.show()


#x_training_in = np.array(list(filter(lambda x: np.abs(x) <= 0.3, x_in)))

#[X_training_1, X_training_2] = np.meshgrid(x_training_in, x_training_in)

#t_training_noise = np.random.normal(0, np.sqrt(sigma2), size=X_training_1.shape) 
#T_training = w_true[0] + (w_true[1] * (X_training_1**2)) + (w_true[2] * (X_training_2**3)) + t_training_noise

#plt.contourf(X_training_1, X_training_2, T_training)
#plt.show()
