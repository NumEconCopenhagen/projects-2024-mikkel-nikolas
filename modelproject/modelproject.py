import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

class SolowModel:
    def __init__(self, s_K=0.2, s_H=0.2, alpha=0.33, phi=0.33, n=0.025, g=0.02, delta=0.05, k0=1.0, h0=1.0, tau_K=0.1, tau_H=0.1):
        # Initialize model parameters
        self.s_K = s_K
        self.s_H = s_H
        self.alpha = alpha
        self.phi = phi
        self.n = n
        self.g = g
        self.delta = delta
        self.k0 = k0
        self.h0 = h0
        self.tau_K = tau_K
        self.tau_H = tau_H
    
    def k_tilde_t(self):
        # Calculate k_tilde_t based on model parameters
        denominator = self.n + self.g + self.delta + self.n * self.g
        k_tilde_t = ((self.s_K ** (1 - self.phi) * self.s_H ** self.phi) / denominator) ** (1 / (1 - self.alpha - self.phi))
        return k_tilde_t

    def h_tilde_t(self):
        # Calculate h_tilde_t based on model parameters
        denominator = self.n + self.g + self.delta + self.n * self.g
        h_tilde_t = ((self.s_K ** self.alpha * self.s_H ** (1 - self.alpha)) / denominator) ** (1 / (1 - self.alpha - self.phi))
        return h_tilde_t

    def y_tilde_t(self):
        # Calculate y_tilde_t based on k_tilde_t and h_tilde_t
        k_tilde_t = self.k_tilde_t()
        h_tilde_t = self.h_tilde_t()
        y_tilde_t = h_tilde_t ** self.phi * k_tilde_t ** self.alpha
        return y_tilde_t

    def obj_k_tilde(self, k):
        # Objective function for k_tilde
        return k - self.k_tilde_t()
    
    def obj_h_tilde(self, h):
        # Objective function for h_tilde
        return h - self.h_tilde_t()
    
    def obj_y_tilde(self, y):
        # Objective function for y_tilde
        return y - self.y_tilde_t()

    def solow_model_dynamics(self, k, h):
        # Calculate next period's k and h based on current values
        k_next = (1 / ((1 + self.n) * (1 + self.g))) * (self.s_K * k**self.alpha * h**self.phi + (1 - self.delta) * k)
        h_next = (1 / ((1 + self.n) * (1 + self.g))) * (self.s_H * k**self.alpha * h**self.phi + (1 - self.delta) * h)
        return k_next, h_next
    
    # Functions for phasediagram without tax
    
    def h_steady_state_delta_k_zero(self, k):
        # Calculate h steady state when delta_k is zero
        return np.minimum(((self.n + self.g + self.delta + self.n*self.g) / self.s_K)**(1 / self.phi) * k**((1 - self.alpha) / self.phi), 100)
    
    def h_steady_state_delta_h_zero(self, k):
        # Calculate h steady state when delta_h is zero
        return np.minimum((self.s_H / (self.n + self.g + self.delta + self.n*self.g))**(1 / (1 - self.phi)) * k**(self.alpha / (1 - self.phi)), 100)
    
    def simulate_solow_model(self, T):
        # Simulate the Solow model for T periods
        k_values = np.zeros(T)
        h_values = np.zeros(T)
        k_values[0] = self.k0
        h_values[0] = self.h0
        
        for t in range(1, T):
            k_next, h_next = self.solow_model_dynamics(k_values[t-1], h_values[t-1])
            k_values[t] = k_next
            h_values[t] = h_next
        
        return k_values, h_values
    
    def ef_k_tilde_t(self):
        # Calculate effective k_tilde_t considering tax rates
        effective_s_K = self.s_K * (1 - self.tau_K)
        effective_s_H = self.s_H * (1 - self.tau_H)
        denominator = self.n + self.g + self.delta + self.n * self.g
        ef_k_tilde_t = ((effective_s_K ** (1 - self.phi) * effective_s_H ** self.phi) / denominator) ** (1 / (1 - self.alpha - self.phi))
        return ef_k_tilde_t

    def ef_h_tilde_t(self):
        # Calculate effective h_tilde_t considering tax rates
        effective_s_K = self.s_K * (1 - self.tau_K)
        effective_s_H = self.s_H * (1 - self.tau_H)
        denominator = self.n + self.g + self.delta + self.n * self.g
        ef_h_tilde_t = ((effective_s_K ** self.alpha * effective_s_H ** (1 - self.alpha)) / denominator) ** (1 / (1 - self.alpha - self.phi))
        return ef_h_tilde_t

    def ef_y_tilde_t(self):
        # Calculate effective y_tilde_t considering tax rates
        ef_k_tilde_t = self.ef_k_tilde_t()
        ef_h_tilde_t = self.ef_h_tilde_t()
        ef_y_tilde_t = ef_h_tilde_t ** self.phi * ef_k_tilde_t ** self.alpha
        return ef_y_tilde_t

    def ef_obj_k_tilde(self, k):
        # Effective objective function for k_tilde considering tax rates
        return k - self.ef_k_tilde_t()
    
    def ef_obj_h_tilde(self, h):
        # Effective objective function for h_tilde considering tax rates
        return h - self.ef_h_tilde_t()
    
    def ef_obj_y_tilde(self, y):
        # Effective objective function for y_tilde considering tax rates
        return y - self.ef_y_tilde_t()

    def ef_solow_model_dynamics(self, k, h):
        # Calculate next period's k and h considering tax rates
        effective_s_K = self.s_K * (1 - self.tau_K)  
        effective_s_H = self.s_H * (1 - self.tau_H)  
        k_next = (1 / ((1 + self.n) * (1 + self.g))) * (effective_s_K * k**self.alpha * h**self.phi + (1 - self.delta) * k)
        h_next = (1 / ((1 + self.n) * (1 + self.g))) * (effective_s_H * k**self.alpha * h**self.phi + (1 - self.delta) * h)
        return k_next, h_next

    
    def simulate_solow_model_no_tax(self, T):
        # Simulate the Solow model for T periods without tax
        k_values = np.zeros(T)
        h_values = np.zeros(T)
        k_values[0] = self.k0
        h_values[0] = self.h0
        
        for t in range(1, T):
            k_next, h_next = self.solow_model_dynamics(k_values[t-1], h_values[t-1])
            k_values[t] = k_next
            h_values[t] = h_next
        
        return k_values, h_values

    # Simulation function with tax
    def ef_simulate_solow_model(self, T):
        # Simulate the Solow model for T periods with tax
        k_values = np.zeros(T)
        h_values = np.zeros(T)
        k_values[0] = self.k0
        h_values[0] = self.h0
        
        for t in range(1, T):
            k_values[t], h_values[t] = self.ef_solow_model_dynamics(k_values[t-1], h_values[t-1])
        
        return k_values, h_values
    
    # Functions for phasediagram with tax
    
    def h_steady_state_delta_k_zero_tax(self, k):
        # Calculate h steady state when delta_k is zero considering tax rates
        effective_s_K = self.s_K * (1 - self.tau_K)
        denominator = self.n + self.g + self.delta + self.n * self.g
        return np.minimum(((denominator / effective_s_K)**(1 / self.phi) * k**((1 - self.alpha) / self.phi)), 100)

    def h_steady_state_delta_h_zero_tax(self, k):
        # Calculate h steady state when delta_h is zero considering tax rates
        effective_s_K = self.s_K * (1 - self.tau_K)
        effective_s_H = self.s_H * (1 - self.tau_H)
        denominator = self.n + self.g + self.delta + self.n * self.g
        return np.minimum((effective_s_H / denominator)**(1 / (1 - self.phi)) * k**(self.alpha / (1 - self.phi)), 100)

