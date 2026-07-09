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
    Initialize the parameters (weights and biases) for a 2-layer neural network.

    Network structure:
        - Input layer with n0 neurons
        - Hidden layer with n1 neurons
        - Output layer with n2 neurons

    All weights and biases are initialized with values drawn from a standard 
    normal distribution (mean = 0, std = 1), which is common for small networks 
    using sigmoid activations.

    Args:
    -----
    n0 : int
        Number of neurons in the input layer (features).
    n1 : int
        Number of neurons in the hidden layer.
    n2 : int
        Number of neurons in the output layer (typically 1 for binary classification).

    Returns:
    --------
    dict
        A dictionary containing all initialized parameters:
            - 'W1': Weight matrix for input → hidden layer (shape: n1 x n0)
            - 'b1': Bias vector for hidden layer (shape: n1 x 1)
            - 'W2': Weight matrix for hidden → output layer (shape: n2 x n1)
            - 'b2': Bias vector for output layer (shape: n2 x 1)
    """

    # ----------------------------
    # Layer 1 (Input → Hidden)
    # ----------------------------

    # Weight matrix W1 has shape (n1, n0)
    # Each row corresponds to one hidden neuron,
    # each column corresponds to one input feature
    W1 = np.random.randn(n1, n0)

    # Bias vector b1 has shape (n1, 1)
    # One bias for each hidden neuron
    b1 = np.random.randn(n1, 1)

    # ----------------------------
    # Layer 2 (Hidden → Output)
    # ----------------------------

    # Weight matrix W2 has shape (n2, n1)
    # Each output neuron connects to all hidden neurons
    W2 = np.random.randn(n2, n1)

    # Bias vector b2 has shape (n2, 1)
    # One bias for each output neuron
    b2 = np.random.randn(n2, 1)

    # ----------------------------
    # Pack all parameters in a dictionary for easy access
    # ----------------------------
    return {
        'W1': W1,  # Weights from input to hidden layer
        'b1': b1,  # Biases for hidden layer
        'W2': W2,  # Weights from hidden to output layer
        'b2': b2   # Biases for output layer
    }
#____________________________________________________________________________
#
def forward_propagation(X, parametres):
    """
    Perform forward propagation through a 2-layer neural network 
    with sigmoid activations on both the hidden and output layers.

    The network structure:
        - Input layer (size: n0)
        - Hidden layer (size: n1)
        - Output layer (size: n2)

    Args:
    -----
    X : numpy.ndarray
        Input data matrix of shape (n_features, n_samples).
        Each column represents one training sample.
        
    parametres : dict
        Dictionary containing the network weights and biases:
            - 'W1': weights from input to hidden layer (shape: n1 x n0)
            - 'b1': biases for hidden layer (shape: n1 x 1)
            - 'W2': weights from hidden to output layer (shape: n2 x n1)
            - 'b2': biases for output layer (shape: n2 x 1)

    Returns:
    --------
    dict
        A dictionary containing the activations of:
            - 'A1': hidden layer after sigmoid activation
            - 'A2': output layer after sigmoid activation (predicted probabilities)
    """

    # ----------------------------
    # Layer 1 (Input → Hidden)
    # ----------------------------

    # Extract weights and biases for layer 1
    W1, b1 = parametres['W1'], parametres['b1']

    # Linear combination: Z1 = W1 · X + b1
    # W1 shape: (n1, n0), X shape: (n0, m), b1 shape: (n1, 1)
    # Result Z1 shape: (n1, m)
    Z1 = W1.dot(X) + b1

    # Apply sigmoid activation to hidden layer
    # A1 shape: (n1, m)
    A1 = 1 / (1 + np.exp(-Z1))

    # ----------------------------
    # Layer 2 (Hidden → Output)
    # ----------------------------

    # Extract weights and biases for layer 2
    W2, b2 = parametres['W2'], parametres['b2']

    # Linear combination: Z2 = W2 · A1 + b2
    # W2 shape: (n2, n1), A1 shape: (n1, m), b2 shape: (n2, 1)
    # Result Z2 shape: (n2, m)
    Z2 = W2.dot(A1) + b2

    # Apply sigmoid activation to output layer
    # A2 shape: (n2, m), values in (0, 1) interpreted as class probabilities
    A2 = 1 / (1 + np.exp(-Z2))

    # Return the activations from both layers for later use (e.g. in backprop)
    return {'A1': A1, 'A2': A2}
#____________________________________________________________________________
#
def back_propagation(X, Y, activations, parametres):
    """
    Perform backpropagation to compute gradients for a 2-layer neural network.

    This function calculates the partial derivatives of the loss function with 
    respect to all weights and biases, based on the chain rule of calculus.

    Args:
    -----
    X : numpy.ndarray
        Input data of shape (n_features, m_samples).
    Y : numpy.ndarray
        Ground truth labels of shape (n_outputs, m_samples), typically binary (0 or 1).
    activations : dict
        Contains forward pass results:
            - 'A1': Activation of the hidden layer (after sigmoid)
            - 'A2': Activation of the output layer (predicted probabilities)
    parametres : dict
        Contains current network parameters:
            - 'W2': Weights from hidden to output layer

    Returns:
    --------
    dict
        Dictionary containing the gradients of weights and biases:
            - 'dW1': Gradient of loss w.r.t W1
            - 'db1': Gradient of loss w.r.t b1
            - 'dW2': Gradient of loss w.r.t W2
            - 'db2': Gradient of loss w.r.t b2
    """

    # ----------------------------
    # Extract Activations and Parameters
    # ----------------------------
    A1 = activations['A1']  # Hidden layer output
    A2 = activations['A2']  # Output layer output (probabilities)
    W2 = parametres['W2']   # Weights from hidden to output layer

    m = Y.shape[1]          # Number of training examples

    # ----------------------------
    # Output Layer Gradients
    # ----------------------------

    # dZ2 = dL/dA2 * dA2/dZ2
    # Since dL/dA2 = (A2 - Y) for binary cross-entropy + sigmoid,
    # and dA2/dZ2 = A2 * (1 - A2), which is already built into the loss derivative
    dZ2 = A2 - Y                            # Shape: (n_outputs, m_samples)

    # Gradient of loss with respect to W2
    dW2 = (1 / m) * dZ2.dot(A1.T)          # Shape: (n_outputs, n_hidden)

    # Gradient of loss with respect to b2
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Shape: (n_outputs, 1)

    # ----------------------------
    # Hidden Layer Gradients
    # ----------------------------

    # Backpropagate the error to the hidden layer:
    # dZ1 = dL/dA1 * dA1/dZ1
    # where dA1/dZ1 = A1 * (1 - A1) (sigmoid derivative)
    dZ1 = W2.T.dot(dZ2) * A1 * (1 - A1)    # Shape: (n_hidden, m_samples)

    # Gradient of loss with respect to W1
    dW1 = (1 / m) * dZ1.dot(X.T)           # Shape: (n_hidden, n_inputs)

    # Gradient of loss with respect to b1
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Shape: (n_hidden, 1)

    # ----------------------------
    # Return all gradients in a dictionary
    # ----------------------------
    return {
        'dW1': dW1,  # Gradient for W1
        'db1': db1,  # Gradient for b1
        'dW2': dW2,  # Gradient for W2
        'db2': db2   # Gradient for b2
    }
#____________________________________________________________________________
#
def update(gradients, parametres, learning_rate):
    """
    Update the parameters of a 2-layer neural network using gradient descent.

    The update is performed using the standard rule:
        parameter = parameter - learning_rate * gradient

    Args:
    -----
    gradients : dict
        Gradients computed during backpropagation:
            - 'dW1', 'db1': gradients for hidden layer weights and biases
            - 'dW2', 'db2': gradients for output layer weights and biases

    parametres : dict
        Current weights and biases of the network:
            - 'W1', 'b1': weights and biases for input → hidden layer
            - 'W2', 'b2': weights and biases for hidden → output layer

    learning_rate : float
        Step size for gradient descent. Determines how much the parameters 
        are adjusted at each iteration.

    Returns:
    --------
    dict
        Updated network parameters after one step of gradient descent.
    """

    # ---------------------------------
    # Extract current parameters
    # ---------------------------------
    W1 = parametres['W1']  # Weights from input to hidden layer
    b1 = parametres['b1']  # Biases for hidden layer
    W2 = parametres['W2']  # Weights from hidden to output layer
    b2 = parametres['b2']  # Biases for output layer

    # ---------------------------------
    # Apply gradient descent updates
    # ---------------------------------

    # Update weights and biases of layer 1 (input → hidden)
    W1 -= learning_rate * gradients['dW1']
    b1 -= learning_rate * gradients['db1']

    # Update weights and biases of layer 2 (hidden → output)
    W2 -= learning_rate * gradients['dW2']
    b2 -= learning_rate * gradients['db2']

    # ---------------------------------
    # Return the updated parameters
    # ---------------------------------
    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
#____________________________________________________________________________
#
def predict(X, parametres):
    """
    Generate binary predictions from a trained 2-layer neural network.

    This function performs a forward pass on the input data using the
    trained weights and biases, then applies a threshold to convert
    predicted probabilities into binary class labels (0 or 1).

    Args:
    -----
    X : numpy.ndarray
        Input data of shape (n_features, n_samples).
        Each column is a feature vector for one sample.

    parametres : dict
        Trained network parameters, including:
            - 'W1', 'b1': weights and biases for input → hidden layer
            - 'W2', 'b2': weights and biases for hidden → output layer

    Returns:
    --------
    numpy.ndarray (bool)
        Binary predictions of shape (1, n_samples), where each element is:
            - True (1) if the output probability ≥ 0.5
            - False (0) otherwise
    """

    # ----------------------------
    # Forward pass through the network
    # ----------------------------

    # Perform forward propagation to get the final output probabilities (A2)
    A2 = forward_propagation(X, parametres)['A2']  # Shape: (1, n_samples)

    # ----------------------------
    # Convert probabilities to binary predictions
    # ----------------------------

    # Apply threshold at 0.5:
    # If predicted probability ≥ 0.5 → predict class 1 (True)
    # Else → predict class 0 (False)
    predictions = A2 >= 0.5

    # Return the binary predictions (as a boolean array)
    return predictions
#____________________________________________________________________________
#
def neural_network(X_train, y_train, n1, learning_rate=0.1, n_iter=1000):
    """
    Train a 2-layer neural network using gradient descent and track training progress.

    The network structure:
        - Input layer with n0 features
        - Hidden layer with n1 neurons (sigmoid activation)
        - Output layer with n2 neurons (typically 1 for binary classification)

    Args:
    -----
    X_train : numpy.ndarray
        Training input data of shape (n_features, m_samples).
    y_train : numpy.ndarray
        Training labels of shape (1, m_samples).
    n1 : int
        Number of neurons in the hidden layer.
    learning_rate : float
        Step size for gradient descent.
    n_iter : int
        Total number of iterations (epochs) to run during training.

    Returns:
    --------
    tuple
        (final trained parameters, training history, meshgrid xx, meshgrid yy)
    """
    # ---------------------------------------------
    # Step 1: Initialization
    # ---------------------------------------------

    n0 = X_train.shape[0]  # Number of input features
    n2 = y_train.shape[0]  # Number of output neurons (usually 1)

    # Initialize weights and biases
    parametres = initialisation(n0, n1, n2)

    # ---------------------------------------------
    # Step 2: Create Meshgrid for Visualization
    # ---------------------------------------------

    # Define plot range for decision boundary visualization
    x_min, x_max = X_train[0, :].min() - 1, X_train[0, :].max() + 1
    y_min, y_max = X_train[1, :].min() - 1, X_train[1, :].max() + 1
    h = 0.01  # Step size for the meshgrid (higher resolution = smoother plots)

    # Generate 2D grid of points for animation of decision boundaries
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    # Flatten and transpose grid points for input to network
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Shape: (2, num_points)

    # ---------------------------------------------
    # Step 3: Initialize Training History
    # ---------------------------------------------

    history = {
        'params': [],       # Store parameter snapshots
        'grid_probs': [],   # Store decision boundary probabilities for animation
        'loss': [],         # Track loss over time
        'acc': []           # Track accuracy over time
    }

    # ---------------------------------------------
    # Step 4: Training Loop
    # ---------------------------------------------
    for i in range(n_iter):

        # 1. Forward pass: compute activations for current parameters
        activations = forward_propagation(X_train, parametres)

        # 2. Backward pass: compute gradients of loss w.r.t. parameters
        gradients = back_propagation(X_train, y_train, activations, parametres)

        # 3. Parameter update: apply gradient descent
        parametres = update(gradients, parametres, learning_rate)

        # 4. Tracking: every 100 iterations or at the final one
        if i % 100 == 0 or i == n_iter - 1:

            # 4a. Predict the output over the mesh grid (for plotting the decision boundary)
            grid_activations = forward_propagation(grid_points, parametres)
            grid_probs = grid_activations['A2'].reshape(xx.shape)
            history['grid_probs'].append(grid_probs)

            # 4b. Save a copy of the parameters (for animation if needed)
            history['params'].append(parametres.copy())

            # 4c. Compute and record current training loss
            loss = log_loss(y_train.flatten(), activations['A2'].flatten())
            history['loss'].append(loss)

            # 4d. Compute and record accuracy on training data
            y_pred = predict(X_train, parametres)
            acc = accuracy_score(y_train.flatten(), y_pred.flatten())
            history['acc'].append(acc)

            print(f"Iteration {i}/{n_iter} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")

    # ---------------------------------------------
    # Step 5: Return Trained Model and Training History
    # ---------------------------------------------
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
#______________________________________________________________________________
#
# Create animation function for real-time display
#______________________________________________________________________________
#
def create_decision_boundary_animation(X, y, history, xx, yy):
    """
    Generate real-time animation of decision boundary evolution during training
    Args:
        X: Input data matrix
        y: True labels vector
        history: Training history containing grid probabilities
        xx: Meshgrid x-coordinates
        yy: Meshgrid y-coordinates
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    plt.show()  # Display animation in real-time

def generate_composite_circles(
    n_samples_outer=100, 
    n_samples_inner=50, 
    outer_radius=0.6, 
    inner_radius=0.2, 
    noise=0.05, 
    distance=None
):

    if distance is None:
        distance = 2 * outer_radius

    def circle_points(n, radius, center_x, center_y, noise):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles) + center_x + np.random.normal(0, noise, size=n)
        y = radius * np.sin(angles) + center_y + np.random.normal(0, noise, size=n)
        return x, y

    # Cerchi esterni (etichetta 0)
    x1, y1 = circle_points(n_samples_outer, outer_radius, 0, 0, noise)
    x2, y2 = circle_points(n_samples_outer, outer_radius, distance, 0, noise)

    # Cerchi interni (etichetta 1)
    x3, y3 = circle_points(n_samples_inner, inner_radius, 0, 0, noise)
    x4, y4 = circle_points(n_samples_inner, inner_radius, distance, 0, noise)

    # Combina punti e etichette
    X = np.vstack((
        np.hstack((x1, x2, x3, x4)),
        np.hstack((y1, y2, y3, y4))
    ))

    y = np.array(
        [0]*(n_samples_outer * 2) + 
        [1]*(n_samples_inner * 2)
    ).reshape(1, -1)

    return X, y

#______________________________________________________________________________
# 
# Modified main execution
#______________________________________________________________________________
#
if __name__ == "__main__":
 
    
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    
    '''
    X, y = generate_composite_circles()
    print(X.shape)
    print(y.shape)
    '''
    
    # Train network and get history
    print("Start training...")
    params, history, xx, yy = neural_network(X, y, n1=10, n_iter=50000, learning_rate=0.01)
    
    # Create and display real-time animation
    create_decision_boundary_animation(X, y, history, xx, yy)
    
    print("Done!")
