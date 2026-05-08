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