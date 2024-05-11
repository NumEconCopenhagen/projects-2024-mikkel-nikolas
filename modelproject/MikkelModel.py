import matplotlib.pyplot as plt
import numpy as np

class ASAD:
    def __init__(self, T, alpha=0.6, gamma=0.08, tol=0.01, z=0, s=0, z_duration=0, s_duration=0):
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.T = T
        self.z = z
        self.s = s
        self.z_duration = z_duration
        self.s_duration = s_duration
        self.delta = 0.97
    
    def solve_model(self):
        self.yhat_vec, self.pihat_vec, self.t_vec = [], [], []
        for t in range(self.T):
            if t == 0 or t <= self.z_duration or t <= self.s_duration:
                z = self.z if t <= self.z_duration else 0
                s = self.s if t <= self.s_duration else 0
                yhat = (z - self.alpha * self.pihat_vec[-1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec[-1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
            else:
                yhat = (0 - self.alpha * self.pihat_vec[-1] - self.alpha * 0) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec[-1] + self.gamma * 0 + 0) / (1 + self.alpha * self.gamma)
            self.yhat_vec.append(yhat)
            self.pihat_vec.append(pihat)
            self.t_vec.append(t)
    
    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.yhat_vec, label="Output gap")
        plt.plot(self.t_vec, self.pihat_vec, label="Inflation gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Output Gap and Inflation Gap over Time")
        plt.legend()
        plt.show()
    
    def plot_ad_as(self):
        y_values = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.pihat_vec

        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="LRAS curve")
        
        for t in range(self.T):
            ad_curve_t = (-1/self.alpha)*(y_values - (self.z if t <= self.z_duration else 0))
            plt.plot(y_values, ad_curve_t, color="blue")

        for t in range(self.T):
            pi_1 = 0 if t == 0 else pi_hat[t-1]
            as_curve_t = pi_1 + self.gamma * y_values + (self.s if t <= self.s_duration else 0)
            plt.plot(y_values, as_curve_t, color="red")

        original_LRAD = (-1/self.alpha)*(y_values - 0)
        plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")

        if self.s_duration == self.T:
            plt.axvline(x=-self.s, color="red", label="LRAS2 curve")
            adjusted_LRAD2 = (-1/self.alpha)*(y_values - (-self.s))
            plt.plot(y_values, adjusted_LRAD2, color="blue", linestyle="--", label="Adjusted LRAD2 curve")

        plt.annotate(r'$\overline{y}$', xy=(-0.0015, -0.0125), fontsize=12)
        plt.annotate(r'$\overline{\pi}$', xy=(-0.0105, -0.0015), fontsize=12)
        plt.xlabel(r'Output gap $(\hat{y})$')
        plt.ylabel(r'Inflation gap $(\hat{\pi})$')
        plt.title(f"Aggregate Demand (AD) and Aggregate Supply (AS)")
        plt.grid()
        plt.legend()
        plt.show()
    
    def solve_stochastic_shock(self, seed):
        self.z_vector, self.s_vector = [], []
        np.random.seed(seed)
        for i in range(self.T):
            if i == 0:
                self.s, self.z = 0, 0
            else:
                self.s = self.s_vector[-1] * 0.15 + np.random.normal(0, 0.2)
                self.z = self.z_vector[-1] * 0.8 + np.random.normal(0, 1)
            self.z_vector.append(self.z)
            self.s_vector.append(self.s)
        
        self.yhat_vec_stoc, self.pihat_vec_stoc, self.t_vec = [], [], []
        for t in range(self.T):
            z = self.z_vector[t]
            s = self.s_vector[t]
            yhat = (z - self.alpha * self.pihat_vec_stoc[-1] - self.alpha * s) / (1 + self.alpha * self.gamma)
            pihat = (self.pihat_vec_stoc[-1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
            self.yhat_vec_stoc.append(yhat)
            self.pihat_vec_stoc.append(pihat)
            self.t_vec.append(t)
    
    def social_loss_function(self):
        social_loss = 0
        for t in range(len(self.yhat_vec_stoc)):
            social_loss += self.delta**t * (self.yhat_vec_stoc[t]**2 + self.pihat_vec_stoc[t]**2)
        return social_loss
    
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