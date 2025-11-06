import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
import matplotlib.pyplot as plt

class GPytorchModel(gpytorch.models.ExactGP):
    """
    A Gaussian Process model using GPytorch with a choice of kernel (Matern 5/2 or RBF).

    Attributes:
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training target data.
        likelihood (gpytorch.likelihoods): Likelihood function.
        kernel_type (str): Type of kernel ('Matern52' or 'RBF').
    """

    def __init__(self, train_x, train_y, likelihood, kernel_type: str = 'Matern52', outputscale: float = None, noise: float = None):
        """
        Initializes the Gaussian Process model with the specified kernel type.

        Args:
            train_x (torch.Tensor): Input training data.
            train_y (torch.Tensor): Target training data.
            likelihood (gpytorch.likelihoods): Gaussian likelihood function.
            kernel_type (str): Kernel type, either 'Matern52' or 'RBF'.
            outputscale (float, optional): if float, fixed and define the hyperparameter outputscale. Default to None.
            noise (float, optional): if float, fixed and define the hyperparameter noise variance. Defaylt to None.
        """
        super().__init__(train_x, train_y, likelihood)
        
        # Define kernel based on 'kernel_type' parameter
        if kernel_type == 'Matern52':
            kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        elif kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        self.mean_module = gpytorch.means.ZeroMean()  # Default mean function is ConstantMean() and not ZeroMean(), ConstantMean() add a hyperparameter to estimate
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)  # Add output scaling factor

        # Register constraints on the covariance module
        self.covar_module.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(0.05, 2.0)
        )
        if outputscale is not None:
            self.covar_module.outputscale = outputscale
            self.covar_module.raw_outputscale.requires_grad = False
        else:
            self.covar_module.register_constraint(
                "raw_outputscale", gpytorch.constraints.Interval(0.5, 3.0)
            )           
        if noise is not None:
            self.likelihood.noise = noise
            self.likelihood.raw_noise.requires_grad = False
        else:
            self.likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.Interval(1e-3, 2)
            )

    def forward(self, x):
        """
        Forward pass for the GP model.

        Args:
            x (torch.Tensor): Input data for prediction.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predictive distribution.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def train_model(self, train_X, train_Y, max_iters=100, lr=0.1, Verbose=False):
        """
        Trains the Gaussian Process model with hyperparameter constraints.

        Args:
            train_X (torch.Tensor): Input training data.
            train_Y (torch.Tensor): Target training data.
            max_iters (int): Maximum number of iterations for optimization.
            lr (float): Learning rate.

        Returns:
            dict: Dictionary containing optimized hyperparameters.
        """
        
        # Optimizer and marginal likelihood
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5, min_lr=0.01
        )

        self.train()
        self.likelihood.train()

        for i in range(max_iters):
            optimizer.zero_grad()
            output = self(train_X)
            loss = -mll(output, train_Y)
            loss.backward()

            if Verbose:
                print(f'Iter {i + 1}/{max_iters} - Loss: {loss.item():.3f}   '
                    f'Lengthscale: {self.covar_module.base_kernel.lengthscale[0].detach().tolist()}   '
                    f'Outputscale: {self.covar_module.outputscale.item():.3f}   '
                    f'Noise: {self.likelihood.noise.item():.3f}    '
                    f'LR: {optimizer.param_groups[0]["lr"]:.3f}')

            optimizer.step()
            scheduler.step(loss)

        return {
            'lengthscale': self.covar_module.base_kernel.lengthscale[0].detach().tolist(),
            'outputscale': self.covar_module.outputscale.detach().item(),
            'noise': self.likelihood.noise.detach().item(),
        }
    
    def get_hyperparameters(self):
        """
        Get the actual (transformed) hyperparameters of the model.
        
        Returns:
            dict: Dictionary containing the current hyperparameters
        """
        return {
            'lengthscale': self.covar_module.base_kernel.lengthscale[0].detach().tolist(),
            'outputscale': self.covar_module.outputscale.detach().item(),
            'noise': self.likelihood.noise.detach().item(),
        }

    def predict(self, test_X):
        """
        Makes predictions on new data.

        Args:
            test_X (torch.Tensor): Input test data.

        Returns:
            tuple: Mean and standard deviation of predictions.
        """ 
        self.eval()
        self.likelihood.eval()

        if test_X.dtype != torch.float64:
            test_X = test_X.double()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self(test_X)
            pred = self.likelihood(output)
        
        self.mean = pred.mean
        self.std = pred.variance.sqrt()

        return self.mean.clone(), self.std.clone()

if __name__ == "__main__":
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Generate example training data with double precision
    train_X = torch.linspace(0, 1, 15, dtype=torch.float64).view(-1, 1)
    train_Y = (torch.sin(train_X * (2 * torch.pi)) + 0.1 * torch.randn_like(train_X))[:, 0]
    train_Y = standardize(train_Y)

    # Instantiate the GP model
    model = GPytorchModel(train_X, train_Y, likelihood, kernel_type='Matern52')

    # Train the model
    optimal_hyperparams = model.train_model(train_X, train_Y, max_iters=100, lr=0.5)

    # Generate test points for evaluation
    test_X = torch.linspace(0, 1, 200, dtype=torch.float64).view(-1, 1)

    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(test_X)

    pred_mean = observed_pred.mean
    pred_std = observed_pred.variance.sqrt()

    # Plotting results
    plt.figure(figsize=(8, 6))
    plt.plot(train_X.numpy(), train_Y.numpy(), 'k*', label='Training Data')
    plt.plot(test_X.numpy(), pred_mean.numpy(), 'b', label='Mean Prediction')
    plt.fill_between(
        test_X.squeeze().numpy(),
        (pred_mean - 1.96 * pred_std).numpy(),
        (pred_mean + 1.96 * pred_std).numpy(),
        color='blue', alpha=0.2, label='Confidence Interval'
    )
    plt.legend()
    plt.title("Gaussian Process Regression")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()
