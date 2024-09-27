# Description: This is a simple demo of variational inference for the Normal-Gamma distribution from Bishop's book.
# written by Richard Xu
# xuyida@hkbu.edu.hk
# Date: 2024-09-23

import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

# clear all memory
plt.close('all')

# Parameters for the Normal-Gamma distribution
alpha_0 = 0.1
beta_0 = 0.1
mu_0 = 0.0
lambda_0 = 1.0

data = np.random.normal(0, 1, 3)
print(data)

mu_axis = np.linspace(-5, 5, 100)
gamma_axis = np.linspace(0.1, 5, 100)
X, Y = np.meshgrid(mu_axis, gamma_axis)
Z = np.zeros_like(X)


# compute the PDF of the normal-gamma distribution

def normal_gamma_pdf(data, mu_val, lambda_val, mu_0, lambda_0, alpha_0, beta_0):
   
    n = len(data)
    mx = np.mean(data)

    # \mu_n & =\frac{\lambda_0 \mu_0 + n \bar{x}}{\lambda_0 + n}
    mu_n = (n * mx + lambda_0 * mu_0 ) / (lambda_0 + n)
    
    #\lambda_n &= \lambda_0 + n
    lambda_n = lambda_0 + n

    #a_n &= a_0 + n/2
    alpha_n = alpha_0 + n / 2

    # b_n &= b_0 + \frac{1}{2} \sum_{i=1}^n (x_i - \bar{x})^2 +  \frac{\lambda_0 n (\bar{x} - \mu_0)^2}{2(\lambda_0 + n)}
    beta_n = beta_0 + 0.5 * np.sum((data - mx)**2) + (lambda_0 * n * (mx - mu_0)**2) / (2 * (lambda_0 + n))
    
    return norm.pdf(mu_val, mu_n, np.sqrt(1 / (lambda_n * lambda_val))) * gamma.pdf(lambda_val, alpha_n,    beta_n, scale=1 / beta_n)




fig, ax = plt.subplots()
plt.xlabel('Mean')
plt.ylabel('Precision')
plt.title('Joint Density of Normal-Gamma Distribution')




# initialize the variational parameter for q(mu) and q(lambda)
num_clicks = 0
mu_n = 3
lambda_n = 0.2
alpha_n = 4
beta_n = 3


def on_click(event):
    
    global num_clicks
    global mu_n
    global lambda_n
    global alpha_n
    global beta_n

    print('num_clicks:', num_clicks)
    print("mu_n:", mu_n)
    print("lambda_n:", lambda_n)
    print("alpha_n:", alpha_n)
    print("beta_n:", beta_n)
    print("**********************")

    if event.button == 1:  # Left mouse button
        
        num_clicks += 1
        ax.clear()
        

        #print(mu_axis, gamma_axis, mu_0, lambda_0, alpha_0, beta_0)
        for i in range(len(mu_axis)):
            for j in range(len(gamma_axis)):
                Z[i, j] = normal_gamma_pdf(data, mu_axis[i], gamma_axis[j], mu_0, lambda_0, alpha_0, beta_0,)
        contour = ax.contour(X, Y, Z, cmap='autumn')



        if num_clicks >1:

            #-----------------------------------------------
            # variational update: only update from second iteration, otherwise, one iteration step looks sufficient
            #-----------------------------------------------      
            n = len(data)
            mx = np.mean(data)
            mu_n = (n * mx + lambda_0 * mu_0 ) / (n + lambda_0)
            lambda_n = (n + lambda_0 ) * (alpha_n / beta_n)
            #----------------------------------------------
            alpha_n = n / 2 + alpha_0
            # E[mu^2] = var + mu^2
            #         = 1 / lambda_n + mu_n^2
            E_mu2 = 1 / lambda_n + mu_n**2
            
            beta_n = beta_0 + 0.5 * ( (n+ lambda_0) * E_mu2  /
                - 2 * ( n * mx + lambda_0 * mu_0 ) * mu_n + sum(data**2)  + lambda_0 * mu_0**2)


        for i in range(len(mu_axis)):
            for j in range(len(gamma_axis)):

                q_mu = norm.pdf(mu_axis[i], mu_n, np.sqrt(1 / (lambda_n)))
                q_lambda = gamma.pdf(gamma_axis[j], alpha_n, beta_n, scale=1 / beta_n)
                Z[i, j] = q_mu * q_lambda
             
        contour = ax.contour(X, Y, Z, cmap='viridis')
    
        plt.xlabel('Mean')
        plt.ylabel('Precision')
        plt.title('Joint Density of Normal-Gamma Distribution')
        print('num_clicks:', num_clicks)
        plt.draw()  # Redraw the plot      

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()




