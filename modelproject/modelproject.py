import matplotlib.pyplot as plt
import numpy as np

class ASAD: 
    # Class for the AS-AD model with initial values for the parameters
    def __init__(self, T, alpha=0.7, gamma=0.075, tol=0.01, z=0, s=0, z_duration=0, s_duration=0): 
        self.alpha = alpha 
        self.gamma = gamma 
        self.tol = tol #tolerance
        self.T = T #number of periods
        self.z = z #shock to aggregate demand
        self.s = s #shock to aggregate supply
        self.z_duration = z_duration #duration of shock to aggregate demand
        self.s_duration = s_duration #duration of shock to aggregate supply
        self.delta = 0.97 #discount factor
        
    def solve_model(self):
        
        """ Solves the model using the initial values for the parameters 
        
        args: 
        self (class): class with initial values for the parameters
        
        Returns:
        self.yhat_vec (list): list of output gaps
        self.pihat_vec (list): list of inflation gaps
        self.t_vec (list): list of periods
        
        """
        # a. Create empty lists
        
        self.yhat_vec = []
        self.pihat_vec = []
        self.t_vec = []
        # b. Solve the model
        
        for t in range(self.T):
            if t == 0:
                yhat = 0
                pihat = 0
                
            elif t <= self.z_duration or t <= self.s_duration:
                z = self.z if t <= self.z_duration else 0
                s = self.s if t <= self.s_duration else 0
                yhat = (z - self.alpha * self.pihat_vec[t - 1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec[t - 1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
                
            else:
                z = 0
                s = 0
                yhat = (z - self.alpha * self.pihat_vec[t - 1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec[t - 1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
            # c. Append to lists
            
            self.yhat_vec.append(yhat)
            self.pihat_vec.append(pihat)
            self.t_vec.append(t)
         
    def plot_results(self): #plots the results
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.yhat_vec, label="Output gap")
        plt.plot(self.t_vec, self.pihat_vec, label="Inflation gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Figure 1: Output gap and Inflation gap")
        plt.legend()
        plt.show()
        
    def plot_ad_as(self):
        
        """ Creating and plotting the AD and AS curves of the given output and inflation gaps
        args:
        self (class): class with initial values for the parameters
        
        Returns: 
        
        original_LRAD (list): list of original LRAD curve
        lras_curve (list): list of LRAS curve
        plot (plot): plot of AD and AS curves

        
        Returns: Temporary shocks:
        
        ad_curve_t (list): list of AD curves
        as_curve_t (list): list of AS curves
        
        Returns: Permenant shocks:
        
        adjusted_LRAD (list): list of adjusted LRAD curve
        LRAS2 (list): list of LRAS2 curve
        
        """
        # a. Create y_values & pi_hat
        
        y_values = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.pihat_vec
        # b. Create functions for AD and AS curves
        
        def ad_function(alpha, y, t, z, z_duration):
            if t <= z_duration:
                z_t = z
            else:
                z_t = 0
            return (-1/alpha)*(y-z_t)

        def as_function(alpha, pi_1, gamma, y, t, s, s_duration):
            
            if t <= s_duration:
                s_t = s
            else:
                s_t = 0
            return pi_1 + gamma * y + s_t

        # Initiate plot
        
        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="LRAS curve") #LRAS curve
        
        # The case for short-term shocks
        if self.s_duration < self.T:
            
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0') #LRAS curve
            for t in range(self.T): #plotting AD curve for each period
                
                ad_curve_t = ad_function(self.alpha, y_values, t, self.z, self.z_duration)
                plt.plot(y_values, ad_curve_t, color="blue")
            # Creating and plotting the SRAS curves
            
            for t in range(self.T):
                if t == 0:
                    pi_1 = 0
                else:
                    pi_1 = pi_hat[t-1]
                    
                as_curve_t = as_function(self.alpha, pi_1, self.gamma, y_values, t, self.s, self.s_duration)
                plt.plot(y_values, as_curve_t, color="red")

            # Original LRAD curve when s_duration < self.T
            
            original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")
        # The case for permanent shocks
        
        if self.s_duration == self.T:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')
            # Original LRAD curve
            
            original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")

            # New LRAS (LRAS2)
            plt.axvline(x=-self.s, color="red", label="LRAS2 curve")

            # Adjusted LRAD2 curve (Central Bank response)
            
            if self.s < 0:  # Positive permanent supply shock
                adjusted_LRAD2 = ad_function(self.alpha, y_values, 0, -self.s, self.z_duration)
                
            else:
                adjusted_LRAD2 = ad_function(self.alpha, y_values, 0, -self.s, self.z_duration)
            plt.plot(y_values, adjusted_LRAD2, color="blue", linestyle="--", label="Adjusted LRAD2 curve")

        # Details for the plot
        
        plt.annotate(r'$\overline{y}$', xy=(-0.0015, -0.0125), fontsize=12)
        plt.annotate(r'$\overline{\pi}$', xy=(-0.0105, -0.0015), fontsize=12)
        plt.xlabel(r'Output gap $(\hat{y})$')
        plt.ylabel(r'Inflation gap $(\hat{\pi})$')
        
        if self.z_duration > 0 and self.z_duration < self.T:
            plt.title(f"Figure 2: {self.z_duration} period positive demand shock")
            
        elif self.z_duration == 0 and self.s_duration == self.T:
            plt.title(f"Figure 2: Permanent supply shock and Central Bank response")
        plt.grid()
        plt.show()
        
    #Stochastic shocks
    def solve_stochastic_shock(self, seed):
        """
        args:
        self (class): class with initial values for the parameters
        seed (int): seed for random number generator

        returns:
        
        self.yhat_vec_stoc (list): list of output gaps
        self.pi_hat_vec_stoc (list): list of inflation gaps
        self.t_vec (list): list of periods
         
        """
        # a. creaat empty lists % setting parameters
        
        self.z_vector = []
        self.s_vector = []
        self.AR_s = 0.15
        self.AR_z = 0.8
        self.simga_s = 0.2
        self.sigma_z = 1
        self.yhat_vec_stoc = []
        self.pihat_vec_stoc = []
        self.t_vec = []
        np.random.seed(seed)
        # b. create z and s vectors
        for i in range(self.T):
            
            if i == 0:
                self.s=0
                self.z=0
                
            else:
                self.s = self.s_vector[-1]*self.AR_s+np.random.normal(0,self.simga_s)
                self.z = self.z_vector[-1]*self.AR_z + np.random.normal(0,self.sigma_z)

            self.z_vector.append(self.z)
            self.s_vector.append(self.s)
        
        # c. create yhat and pihat vectors
                
        for t in range(self.T):
            
            if t == 0:
                yhat = 0
                pihat = 0
                        
            else:
                z = self.z_vector[t]
                s = self.s_vector[t]
                yhat = (z - self.alpha * self.pihat_vec_stoc[t - 1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec_stoc[t - 1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)

            # e. append to empty lists
            self.yhat_vec_stoc.append(yhat)
            self.pihat_vec_stoc.append(pihat)
            self.t_vec.append(t)

    # creating social loss function & calculating social loss   
    def social_loss_function(self):
        social_loss = 0
        for t in range(len(self.yhat_vec_stoc)): # Loop through all periods as the discount factor is correctly applied
            social_loss += self.delta**t * (self.yhat_vec_stoc[t]**2 + self.pihat_vec_stoc[t]**2)
        return social_loss
    
    # social loss function with alpha as input
    def social_loss_alpha(self, alpha_value, seed):
        self.alpha = alpha_value # Update the alpha value
        self.solve_model() # Solve the model
        self.solve_stochastic_shock(seed) # Solve the model with stochastic shock
        return self.social_loss_function() # Return the social loss function
    
    def average_social_loss(alpha_value, n_simulations=500): # Average social loss function
        social_losses = [] # Empty list for social losses
        for i in range(n_simulations): # Loop through all simulations
            seed = i # Set seed
            asad_model = ASAD(T=100, alpha=alpha_value ,gamma=0.075) # Create model
            asad_model.solve_model() # Solve model
            asad_model.solve_stochastic_shock(seed) # Solve model with stochastic shock
            social_losses.append(asad_model.social_loss_function()) # Append social loss to list
        return np.mean(social_losses) # Return average social loss
