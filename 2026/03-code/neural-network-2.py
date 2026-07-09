#______________________________________________________________________________
# Neural Network for Binary Classification
# 
# This code implements a 2-layer neural network (1 hidden layer) to classify
# synthetic circular data. It includes visualization of training metrics and
# decision boundaries.
# 
# Structure:
# - Input layer (n0 nodes) -> Hidden layer (n1 nodes) -> Output layer (n2 nodes)
# - Sigmoid activation for both layers
# - Backpropagation with gradient descent optimization
#______________________________________________________________________________
#
# Import Libraries
#______________________________________________________________________________
#
import numpy as np                  # Numerical computing
import matplotlib.pyplot as plt     # Plotting
import matplotlib.animation as animation  # For animations (not used in current code)

from sklearn.metrics import log_loss       # Loss calculation
from sklearn.datasets import make_blobs    # Dataset generation (not used)
from sklearn.metrics import accuracy_score # Accuracy calculation
from sklearn.datasets import make_circles  # Circular dataset generation

#____________________________________________________________________________
#
def initialisation(n0, n1, n2):
    """
    Initialize neural network parameters with random values
    Args:
        n0: Input layer size
        n1: Hidden layer size
        n2: Output layer size
    Returns:
        Dictionary containing initialized weights and biases
    """
    # Initialize weights with random normal distribution values
    W1 = np.random.randn(n1, n0)  # Input to hidden layer weights
    b1 = np.random.randn(n1, 1)   # Hidden layer biases
    W2 = np.random.randn(n2, n1)  # Hidden to output layer weights
    b2 = np.random.randn(n2, 1)   # Output layer biases

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

#____________________________________________________________________________
#
def forward_propagation(X, parametres):
    """
    Perform forward propagation through the network
    Args:
        X: Input data (features)
        parametres: Dictionary containing network parameters
    Returns:
        Dictionary containing layer activations
    """
    # Extract parameters
    W1, b1 = parametres['W1'], parametres['b1']
    W2, b2 = parametres['W2'], parametres['b2']

    # Hidden layer calculations
    Z1 = W1.dot(X) + b1           # Linear combination
    A1 = 1 / (1 + np.exp(-Z1))    # Sigmoid activation

    # Output layer calculations
    Z2 = W2.dot(A1) + b2          # Linear combination
    A2 = 1 / (1 + np.exp(-Z2))    # Sigmoid activation

    return {'A1': A1, 'A2': A2}
#____________________________________________________________________________
#
def back_propagation(X, Y, activations, parametres):
    """
    Perform backward propagation to calculate gradients
    Args:
        X: Input data
        Y: True labels
        activations: Dictionary from forward propagation
        parametres: Dictionary containing network parameters
    Returns:
        Dictionary containing parameter gradients
    """
    # Extract activations and parameters
    A1, A2 = activations['A1'], activations['A2']
    W2 = parametres['W2']
    m = Y.shape[1]  # Number of examples

    # Output layer gradients
    dZ2 = A2 - Y                    # Derivative of loss w.r.t Z2
    dW2 = (1/m) * dZ2.dot(A1.T)     # Gradient for W2
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # Gradient for b2

    # Hidden layer gradients
    dZ1 = W2.T.dot(dZ2) * A1 * (1 - A1)  # Derivative w.r.t Z1
    dW1 = (1/m) * dZ1.dot(X.T)      # Gradient for W1
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # Gradient for b1

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
#____________________________________________________________________________
#
def update(gradients, parametres, learning_rate):
    """
    Update parameters using gradient descent
    Args:
        gradients: Dictionary from backpropagation
        parametres: Current network parameters
        learning_rate: Optimization step size
    Returns:
        Updated parameters dictionary
    """
    # Extract parameters
    W1, b1 = parametres['W1'], parametres['b1']
    W2, b2 = parametres['W2'], parametres['b2']

    # Update parameters using gradients
    W1 -= learning_rate * gradients['dW1']
    b1 -= learning_rate * gradients['db1']
    W2 -= learning_rate * gradients['dW2']
    b2 -= learning_rate * gradients['db2']

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

#____________________________________________________________________________
#
def predict(X, parametres):
    """
    Make binary predictions using trained network
    Args:
        X: Input data
        parametres: Trained network parameters
    Returns:
        Binary predictions (0 or 1)
    """
    # Forward pass and threshold at 0.5
    A2 = forward_propagation(X, parametres)['A2']
    return A2 >= 0.5
#____________________________________________________________________________
#
def neural_network(X_train, y_train, n1, learning_rate=0.1, n_iter=1000):
    """
    Train neural network and track training history
    Args:
        X_train: Training data (shape: n_features x m_samples)
        y_train: Training labels (shape: 1 x m_samples)
        n1: Number of hidden neurons
        learning_rate: Gradient descent step size
        n_iter: Number of training iterations
    Returns:
        tuple: (final parameters, training history, meshgrid coordinates)
    """
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    parametres = initialisation(n0, n1, n2)
    
    # Create grid for animation frames
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    
    # Store history for animation
    history = {
        'params': [],
        'grid_probs': [],
        'loss': [],
        'acc': []
    }
    
    for i in range(n_iter):
        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)
        
        # Capture every 100 iterations AND the final iteration
        if i % 100 == 0 or i == n_iter - 1:
            # Calculate grid probabilities
            grid_activations = forward_propagation(grid_points, parametres)
            history['grid_probs'].append(grid_activations['A2'].reshape(xx.shape))
            history['params'].append(parametres.copy())
            
            # Store metrics
            history['loss'].append(log_loss(y_train, activations['A2']))
            y_pred = predict(X_train, parametres)
            history['acc'].append(accuracy_score(y_train.flatten(), y_pred.flatten()))
    
    return parametres, history, xx, yy
#____________________________________________________________________________
#
def plot_decision_boundary(X, y, parametres):
    """
    Plot decision boundary of trained network with custom styling
    Args:
        X: Input data
        y: True labels
        parametres: Trained network parameters
    """
    # Create meshgrid for contour plot
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # Step size for meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T

    # Get probability predictions (not thresholded)
    activations = forward_propagation(grid_points, parametres)
    probabilities = activations['A2'].reshape(xx.shape)

    # Create plot with custom styling
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Set background color to white
    ax.set_facecolor('white')
    
    # Plot filled contours with transparency
    contour = plt.contourf(xx, yy, probabilities, 
                          levels=50, cmap='bwr', alpha=0.2)
    
    # Plot decision boundary line in orange
    boundary = plt.contour(xx, yy, probabilities, 
                          levels=[0.5], colors='orange', linewidths=2)
    
    # Plot original data points
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), 
               cmap='coolwarm', edgecolors='k', s=50)
    
    # Add labels and title
    plt.title("Decision Boundary with Orange Separation Line", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    
    # Add colorbar for probability
    plt.colorbar(contour, label='Class Probability')
    
    plt.show()
#____________________________________________________________________________
# 
# Create animation function
def create_decision_boundary_animation(X, y, history, xx, yy):
    """
    Generate animation of decision boundary evolution during training
    Args:
        X: Input data matrix
        y: True labels vector
        history: Training history containing grid probabilities
        xx: Meshgrid x-coordinates
        yy: Meshgrid y-coordinates
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    #-----------------------------------------------------------    
    def update(frame):
        """Update function for animation frames"""
        ax.clear()
        ax.set_facecolor('white')
        
        # Plot decision boundary
        probabilities = history['grid_probs'][frame]
        ax.contourf(xx, yy, probabilities, levels=50, cmap='bwr', alpha=0.2)
        boundary = ax.contour(xx, yy, probabilities, levels=[0.5], colors='orange', linewidths=2)
        ax.scatter(X[0, :], X[1, :], c=y.flatten(), cmap='coolwarm', edgecolors='k', s=50)
        
        # Add iteration info
        iteration = (frame + 1) * 100
        ax.set_title(f"Iteration: {iteration}\nLoss: {history['loss'][frame]:.4f}  Accuracy: {history['acc'][frame]:.2f}")
        
        return boundary

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(history['grid_probs']),
        interval=100,
        blit=False
    )
    
    plt.close()
    return ani
#____________________________________________________________________________
# 
# Modified main execution
if __name__ == "__main__":
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    # Train network and get history
    params, history, xx, yy = neural_network(X, y, n1=10, n_iter=50000, learning_rate=0.01)
    
    # Create and display animation
    ani = create_decision_boundary_animation(X, y, history, xx, yy)
    
    # To save the animation (requires ffmpeg)
    ani.save('decision_boundary_evolution_1.mp4', writer='ffmpeg', fps=10)
    
    # To display in Jupyter notebook
    #from IPython.display import HTML
    #HTML(ani.to_html5_video())
    
    print("Done!")