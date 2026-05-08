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

#3)

# designs matrix Φ(x) = [1, x1^2, x2^3]
def design_matrix(x1, x2):
    return np.column_stack((np.ones(len(x1)), x1**2, x2**3))

phi_training = design_matrix(x1_training, x2_training)
print(phi_training)
phi_test = design_matrix(x1_test, x2_test)
print(phi_test)

# maximum likelihood estimation of w (eq.19) w_ML = (Φ^T Φ)^(-1) Φ^T t
w_ML = np.linalg.inv(phi_training.T @ phi_training) @ phi_training.T @ t_training
print(w_ML)

# maximum likelihood estimation of beta (eq.20) beta_ML = (1/N) * sum(ti - wML^T phi(xi))^2
residuals_training = t_training - (phi_training @ w_ML)
beta_ML = len(t_training) / np.sum(residuals_training**2)
print(beta_ML)

print(f"w_ML:    {w_ML}")
print(f"w_true:  {w_true}")
print(f"beta_ML: {beta_ML:.4f}  (true beta = {1/sigma2:.4f})")

# prediction for test data t_pred_ML = phi_test @ w_ML
t_pred_ML = phi_test @ w_ML
print(t_pred_ML)

MSE_test = np.mean((t_pred_ML - t_test)**2)
print(f"MSE (Test): {MSE_test:.4f}")

plt.scatter(x1_test, x2_test, c=t_pred_ML, cmap='viridis')
plt.title(f"Predicted t (ML) for test data (MSE = {MSE_test:.4f})")
plt.show()
# plot predicted vs actual test values
plt.scatter(t_test, t_pred_ML, alpha=0.5)
plt.plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 'r--', label='Perfect prediction')
plt.xlabel('Actual t (test)')
plt.ylabel('Predicted t (ML)')
plt.title(f"ML predictions vs actual test values (MSE = {MSE_test:.4f})")
plt.legend()
plt.show()

#4)

beta = 1/sigma2

# higher alpha means stronger prior 'belief' in w being close to zero, lower alpha means weaker prior
# higher -> worse MSE but lower predictive variance, lower -> better MSE but higher predictive variance
alphas = [0.2, 0.8, 2.0] # multiple different values for comparison and experimentation
 
# for-loop to test different alphas simultaneously, does not need to kept but useful //Edvard
for alpha in alphas:
    # posterior over w
    S_N = np.linalg.inv(alpha * np.eye(3) + beta * (phi_training.T @ phi_training))   # Eq. 26
    m_N = beta * S_N @ phi_training.T @ t_training                                    # Eq. 25
 
    # predictive distribution for test points
    # mu_N: one mean per test point
    # sigma2_N: one variance per test point
    mu_N_test     = phi_test @ m_N                                                    # Eq. 30
    sigma2_N_test = (1/beta) + np.diag(phi_test @ S_N @ phi_test.T)                   # Eq. 31
 
    MSE_bayes = np.mean((mu_N_test - t_test)**2)
    print(f"alpha={alpha}: Bayes Test MSE = {MSE_bayes:.4f},  " f"mean predictive variance = {np.mean(sigma2_N_test):.4f}")
 
    # plot predicted vs actual
    plt.scatter(t_test, mu_N_test, alpha=0.5, label=f'alpha={alpha}')
 
plt.plot([t_test.min(), t_test.max()], [t_test.min(), t_test.max()], 'r--', label='Perfect prediction')
plt.xlabel('Actual t (test)')
plt.ylabel('Predicted t (Bayesian mean)')
plt.title("Bayesian predictions vs actual test values (all alphas)")
plt.legend()
plt.show()