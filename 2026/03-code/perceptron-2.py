import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_blobs

np.random.seed(42)

# Generate a linearly separable dataset using make_blobs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=1.0, random_state=1234)

# Define a modified Perceptron class that stores history for animation
class Perceptron:
    """
    A simple Perceptron for binary classification that stores its weight history
    to visualize the convergence process.
    """
    def __init__(self, input_size, learning_rate=0.01, epochs=50):
        """
        Initialize the perceptron with zeros for weights and a zero bias.
        
        Parameters:
            input_size (int): Number of features.
            learning_rate (float): The learning rate.
            epochs (int): Number of epochs (complete passes through the dataset).
        """
        self.weights         = np.random.randn(input_size)
        self.bias            = np.random.randn()
        self.learning_rate   = learning_rate
        self.epochs          = epochs
        self.error_threshold = 1e-14
        # History will store (weights, bias) at each epoch
        self.history = []

    def activate(self, x):
        """
        Step activation function.
        
        Parameters:
            x (float): The linear combination result.
        
        Returns:
            int: Returns 1 if x >= 0, otherwise 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, x):
        """
        Compute the prediction for a given input.
        
        Parameters:
            x (numpy.ndarray): Input vector.
            
        Returns:
            int: Predicted class (0 or 1).
        """
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activate(linear_output)

    def train(self, X, y):
        """
        Train the perceptron on dataset X with labels y.
        
        This method updates the weights over a fixed number of epochs.
        At the end of each epoch, the current weights and bias are stored
        in the history attribute for visualization.
        
        Parameters:
            X (numpy.ndarray): Input dataset, each row is a sample.
            y (numpy.ndarray): Corresponding labels.
        """
        # Loop over each epoch
        for epoch in range(self.epochs):
            # Shuffle data to better simulate a stochastic update process
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            total_error = 0  # Track cumulative error for the epoch
        
            for i in indices:
                prediction = self.predict(X[i])
                error = y[i] - prediction
                total_error += abs(error)  # Accumulate absolute error
        
                # Update rule for the weights and bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
        
            # Store a copy of the current weights and bias for this epoch
            self.history.append((self.weights.copy(), self.bias))
        
            # Check if the total error is below the threshold
            if total_error < self.error_threshold:
                print(f"Stopping early at epoch {epoch} with total error {total_error:.4f}")
                break
            
    def evaluate(self, X, y):
        """
        Evaluate the perceptron on dataset X with labels y.
        
        Returns:
            float: The accuracy (fraction of correctly classified samples).
        """
        predictions = [self.predict(x) for x in X]
        accuracy = np.mean(np.array(predictions) == y)
        return accuracy

# Initialize and train the perceptron
input_size = X.shape[1]
perceptron = Perceptron(input_size=input_size, learning_rate=0.0005, epochs=500)
perceptron.train(X, y)
accuracy = perceptron.evaluate(X, y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# Set up the plot for the animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Create a line object for the decision boundary that will be updated
decision_boundary_line, = ax.plot([], [], 'k--', lw=2, label='Decision Boundary')
epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8))
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

def init():
    """
    Initialization function for the animation.
    Clears the decision boundary and epoch text.
    """
    decision_boundary_line.set_data([], [])
    epoch_text.set_text('')
    return decision_boundary_line, epoch_text

def update(frame):
    """
    Update function for the animation.
    
    This function is called for each frame (each epoch).
    It retrieves the weights and bias from the training history and
    updates the decision boundary line accordingly.
    
    Parameters:
        frame (int): The current epoch index.
    """
    # Get the current weights and bias from the stored history
    w, b = perceptron.history[frame]
    
    # Compute x-values for the decision boundary line
    x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
    
    # Compute the corresponding y-values for the decision boundary:
    # The decision boundary is defined by: w[0]*x + w[1]*y + b = 0
    # Solve for y: y = -(w[0] * x + b) / w[1]
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        # If w[1] is zero, the line is vertical; plot a vertical line.
        x_vals = np.full_like(x_vals, -b / w[0])
        y_vals = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 200)
    
    # Update the line data with the new decision boundary
    decision_boundary_line.set_data(x_vals, y_vals)
    
    # Update the text to display the current epoch number
    epoch_text.set_text(f'Epoch: {frame + 1}/{perceptron.epochs}')
    
    return decision_boundary_line, epoch_text

# Create the animation using FuncAnimation.
# The 'frames' parameter is set to the number of stored epochs in perceptron.history.
ani = animation.FuncAnimation(fig, update, frames=len(perceptron.history),
                              init_func=init, interval=200, blit=True)

# If running in a Jupyter Notebook, you might need to use:
# %matplotlib notebook
# or
# %matplotlib qt
# to enable interactive animations.

plt.show()
