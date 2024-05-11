import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

class ASAD: 
    def __init__(self, T, beta=0.5, gamma=0.07, tol=0.02, z=0, s=0, z_duration=0, s_duration=0): 
        self.beta = beta 
        self.gamma = gamma 
        self.tol = tol  # tolerance
        self.T = T  # number of periods
        self.z = z  # shock to aggregate demand
        self.s = s  # shock to aggregate supply
        self.z_duration = z_duration  # duration of shock to aggregate demand
        self.s_duration = s_duration  # duration of shock to aggregate supply
        self.delta = 0.95  # discount factor
        
    def solve_model(self):
        # Create empty lists
        self.yhat_vec = []
        self.pihat_vec = []
        self.t_vec = []
        self.er_vec = []  # Initialize er_vec list

        z = 0  # Initialize z outside the conditional blocks
        s = 0  # Initialize s outside the conditional blocks
        
        for t in range(self.T):
            if t == 0:
                yhat = 0
                pihat = 0
                er = 0  # Set initial value of er to 0
            elif t <= self.z_duration or t <= self.s_duration:
                z = self.z if t <= self.z_duration else 0
                s = self.s if t <= self.s_duration else 0
                yhat = (z - self.beta * self.er_vec[t - 1] - self.beta * self.pihat_vec[t - 1] - self.beta * s) / (1 + self.beta * self.gamma)
                pihat = (self.pihat_vec[t - 1] + self.er_vec[t - 1] + self.gamma * z + s) / (1 + self.beta * self.gamma)
                er = (yhat - z) / self.beta  # Calculate er
            else:
                z = 0
                s = 0
                yhat = (z - self.beta * self.er_vec[t - 1] - self.beta * self.pihat_vec[t - 1] - self.beta * s) / (1 + self.beta * self.gamma)
                pihat = (self.pihat_vec[t - 1] + self.er_vec[t - 1] + self.gamma * z + s) / (1 + self.beta * self.gamma)
                er = (yhat - z) / self.beta  # Calculate er
            
            # Append calculated er to er_vec list
            self.er_vec.append(er)
            
            # Append yhat, pihat, and t to respective lists
            self.yhat_vec.append(yhat)
            self.pihat_vec.append(pihat)
            self.t_vec.append(t)

    def plot_results(self): 
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.yhat_vec, label="Output-gap")
        plt.plot(self.t_vec, self.pihat_vec, label="Inflation-gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Figure 1: Output-gap and Inflation-gap")
        plt.legend()
        plt.show()
        
    def plot_ad_as(self):
        y_values = np.linspace(-0.02, 0.02, 100)
        pi_hat = self.pihat_vec

        def ad_function(beta, y, t, z, z_duration):
            z_t = z if t <= z_duration else 0
            return (-1 / beta) * (y - z_t)

        def as_function(beta, pi_1, gamma, y, t, s, s_duration):
            s_t = s if t <= s_duration else 0
            return pi_1 + gamma * y + s_t

        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="blue", label="LRAS-curve") 

        if self.s_duration < self.T:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation-Gap = 0')           
            for t in range(self.T):
                ad_curve_t = ad_function(self.beta, y_values, t, self.z, self.z_duration)
                plt.plot(y_values, ad_curve_t, color="green")

            for t in range(self.T):
                pi_1 = 0 if t == 0 else pi_hat[t - 1]
                as_curve_t = as_function(self.beta, pi_1, self.gamma, y_values, t, self.s, self.s_duration)
                plt.plot(y_values, as_curve_t, color="blue")       
            original_LRAD = ad_function(self.beta, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="green", label="Original LRAD-curve")

        if self.s_duration == self.T:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation-gap = 0')
            original_LRAD = ad_function(self.beta, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")
            plt.axvline(x=-self.s, color="red", label="LRAS2 curve")

            adjusted_LRAD2 = ad_function(self.beta, y_values, 0, -self.s, self.z_duration)
            plt.plot(y_values, adjusted_LRAD2, color="blue", linestyle="--", label="Adjusted LRAD2 curve")

        plt.annotate(r'$\overline{y}$', xy=(-0.0015, -0.0125), fontsize=12)
        plt.annotate(r'$\overline{\pi}$', xy=(-0.0105, -0.0015), fontsize=12)
        plt.xlabel(r'Output-gap $(\hat{y})$')
        plt.ylabel(r'Inflation-gap $(\hat{\pi})$')
        
        if 0 < self.z_duration < self.T:
            plt.title(f"Figure 2: {self.z_duration} period positive demand shock")
        elif self.z_duration == 0 and self.s_duration == self.T:
            plt.title(f"Figure 2: Permanent supply shock and Central Bank response")
        plt.grid()
        plt.show()

    def solve_stochastic_shock(self, seed): 
        self.z_vector = []
        self.s_vector = []
        self.AR_s = 0.15
        self.AR_z = 0.8
        self.sigma_s = 0.2
        self.sigma_z = 1
        self.yhat_vec_stoc = []
        self.pihat_vec_stoc = []
        self.t_vec = []
        np.random.seed(seed)

