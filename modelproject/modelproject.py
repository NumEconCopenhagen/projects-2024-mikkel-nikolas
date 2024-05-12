import matplotlib.pyplot as plt
import numpy as np

class ASAD:
    def __init__(self, T, alpha=0.7, gamma=0.075, tol=0.01, z=0, s=0, z_duration=0, s_duration=0):
        self.alpha, self.gamma, self.tol, self.T = alpha, gamma, tol, T
        self.z, self.s, self.z_duration, self.s_duration = z, s, z_duration, s_duration
        self.delta = 0.97

    def solve_model(self):
        self.yhat_vec, self.pihat_vec, self.t_vec = [], [], []
        for t in range(self.T):
            if t == 0:
                yhat, pihat = 0, 0
            else:
                z = self.z if t <= self.z_duration else 0
                s = self.s if t <= self.s_duration else 0
                yhat = (z - self.alpha * self.pihat_vec[t - 1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec[t - 1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
            self.yhat_vec.append(yhat)
            self.pihat_vec.append(pihat)
            self.t_vec.append(t)

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.yhat_vec, label="Output gap")
        plt.plot(self.t_vec, self.pihat_vec, label="Inflation gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Figure 1: Output gap and Inflation gap")
        plt.legend()
        plt.show()

    def plot_ad_as(self):
        y_values = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.pihat_vec

        def ad_function(alpha, y, t, z, z_duration):
            z_t = z if t <= z_duration else 0
            return (-1/alpha)*(y - z_t)

        def as_function(alpha, pi_1, gamma, y, t, s, s_duration):
            s_t = s if t <= s_duration else 0
            return pi_1 + gamma * y + s_t

        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="LRAS curve")

        for t in range(self.T):
            ad_curve_t = ad_function(self.alpha, y_values, t, self.z, self.z_duration)
            plt.plot(y_values, ad_curve_t, color="blue")

        for t in range(self.T):
            pi_1 = 0 if t == 0 else pi_hat[t-1]
            as_curve_t = as_function(self.alpha, pi_1, self.gamma, y_values, t, self.s, self.s_duration)
            plt.plot(y_values, as_curve_t, color="red")

        original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
        plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")

        plt.xlabel(r'Output gap $(\hat{y})$')
        plt.ylabel(r'Inflation gap $(\hat{\pi})$')

        if 0 < self.z_duration < self.T:
            plt.title(f"Figure 2: {self.z_duration} period positive demand shock")
        elif self.z_duration == 0 and self.s_duration == self.T:
            plt.title(f"Figure 2: Permanent supply shock and Central Bank response")

        plt.grid()
        plt.show()

    def solve_stochastic_shock(self, seed):
        self.z_vector, self.s_vector = [], []
        self.AR_s, self.AR_z = 0.15, 0.8
        self.sigma_s, self.sigma_z = 0.2, 1
        self.yhat_vec_stoc, self.pihat_vec_stoc, self.t_vec = [], [], []

        np.random.seed(seed)
        for i in range(self.T):
            self.s = 0 if i == 0 else self.s_vector[-1] * self.AR_s + np.random.normal(0, self.sigma_s)
            self.z = 0 if i == 0 else self.z_vector[-1] * self.AR_z + np.random.normal(0, self.sigma_z)
            self.z_vector.append(self.z)
            self.s_vector.append(self.s)

        for t in range(self.T):
            if t == 0:
                yhat, pihat = 0, 0
            else:
                z, s = self.z_vector[t], self.s_vector[t]
                yhat = (z - self.alpha * self.pihat_vec_stoc[t - 1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec_stoc[t - 1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
            self.yhat_vec_stoc.append(yhat)
            self.pihat_vec_stoc.append(pihat)
            self.t_vec.append(t)

    def social_loss_function(self):
        social_loss = 0
        for t in range(len(self.yhat_vec_stoc)): # Loop through all periods as the discount factor is correctly applied
            social_loss += self.delta**t * (self.yhat_vec_stoc[t]**2 + self.pihat_vec_stoc[t]**2)
        return social_loss
    
    def social_loss_alpha(self, alpha_value, seed):
        self.alpha = alpha_value 
        self.solve_model() 
        self.solve_stochastic_shock(seed) # Solve the model with stochastic shock
        return self.social_loss_function() 
    

    @staticmethod
    def average_social_loss(alpha_value, n_simulations=500):
        social_losses = []
        for i in range(n_simulations):
            seed = i
            asad_model = ASAD(T=100, alpha=alpha_value, gamma=0.075)
            asad_model.solve_model()
            asad_model.solve_stochastic_shock(seed)
            social_losses.append(asad_model.social_loss_function())
        return np.mean(social_losses)
