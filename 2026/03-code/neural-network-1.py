#______________________________________________________________________________
#
# Import Libraries
#______________________________________________________________________________
#
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.animation as animation

from sklearn.metrics  import log_loss
from sklearn.datasets import make_blobs
from sklearn.metrics  import accuracy_score
#____________________________________________________________________________
#
def initialisation(n0, n1, n2):
    """
    Initialize the weights and biases for a simple 2-layer neural network.

    The network structure is:
        - Input layer with n0 neurons
        - Hidden layer with n1 neurons
        - Output layer with n2 neurons

    Parameters:
    -----------
    n0 : int
        Number of neurons in the input layer.
    n1 : int
        Number of neurons in the hidden layer.
    n2 : int
        Number of neurons in the output layer.

    Returns:
    --------
    parametres : dict
        A dictionary containing the initialized weights and biases:
            - 'W1': weights for layer 1, shape (n1, n0)
            - 'b1': biases for layer 1, shape (n1, 1)
            - 'W2': weights for layer 2, shape (n2, n1)
            - 'b2': biases for layer 2, shape (n2, 1)
    """
    # Randomly initialize weights for the first layer (input → hidden)
    # Shape: (number of neurons in hidden layer, number of inputs)
    W1 = np.random.randn(n1, n0) 

    # Randomly initialize biases for the first layer
    # Shape: (number of neurons in hidden layer, 1)
    b1 = np.random.randn(n1, 1)

    # Randomly initialize weights for the second layer (hidden → output)
    # Shape: (number of neurons in output layer, number of hidden neurons)
    W2 = np.random.randn(n2, n1) 

    # Randomly initialize biases for the second layer
    # Shape: (number of neurons in output layer, 1)
    b2 = np.random.randn(n2, 1)

    # Store all parameters in a dictionary for convenience
    parametres = {
        'W1': W1,  # Weights from input to hidden
        'b1': b1,  # Biases for hidden layer
        'W2': W2,  # Weights from hidden to output
        'b2': b2   # Biases for output layer
    }

    # Return the initialized parameters
    return parametres
#____________________________________________________________________________
#
def forward_propagation(X, parametres):
    """
    Perform forward propagation for a 2-layer neural network 
    with sigmoid activation functions.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data matrix of shape (n_features, n_samples).
    parametres : dict
        Dictionary containing the weights and biases:
            - 'W1': weights for layer 1, shape (n1, n0)
            - 'b1': biases for layer 1, shape (n1, 1)
            - 'W2': weights for layer 2, shape (n2, n1)
            - 'b2': biases for layer 2, shape (n2, 1)

    Returns:
    --------
    activations : dict
        Dictionary containing the activations of each layer:
            - 'A1': activation of hidden layer, shape (n1, n_samples)
            - 'A2': activation of output layer, shape (n2, n_samples)
    """
    # Extract weights and biases from the parameter dictionary
    W1 = parametres['W1']  # Weights from input to hidden layer
    b1 = parametres['b1']  # Biases for hidden layer
    W2 = parametres['W2']  # Weights from hidden to output layer
    b2 = parametres['b2']  # Biases for output layer

    # Compute the linear combination for the hidden layer
    # Z1 shape: (n1, n_samples)
    Z1 = W1.dot(X) + b1

    # Apply sigmoid activation function to hidden layer
    # A1 shape: (n1, n_samples)
    A1 = 1 / (1 + np.exp(-Z1))

    # Compute the linear combination for the output layer
    # Z2 shape: (n2, n_samples)
    Z2 = W2.dot(A1) + b2

    # Apply sigmoid activation function to output layer
    # A2 shape: (n2, n_samples), represents predicted probabilities
    A2 = 1 / (1 + np.exp(-Z2))

    # Store activations in a dictionary for later use (e.g., in backpropagation)
    activations = {
        'A1': A1,  # Activation from hidden layer
        'A2': A2   # Final prediction (output layer)
    }

    # Return the dictionary of activations
    return activations
#____________________________________________________________________________
#
def back_propagation(X, Y, activations, parametres):
    """
    Perform backpropagation to compute gradients for a 2-layer neural network.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (n_features, n_samples).
    Y : numpy.ndarray
        True labels of shape (n_outputs, n_samples), typically binary (0 or 1).
    activations : dict
        Dictionary containing forward activations:
            - 'A1': activation of the hidden layer
            - 'A2': activation of the output layer (predictions)
    parametres : dict
        Dictionary containing weights and biases:
            - 'W2': weights from hidden to output layer

    Returns:
    --------
    gradients : dict
        Dictionary of computed gradients:
            - 'dW1': gradient of loss with respect to W1
            - 'db1': gradient of loss with respect to b1
            - 'dW2': gradient of loss with respect to W2
            - 'db2': gradient of loss with respect to b2
    """

    # Retrieve activations from forward propagation
    A1 = activations['A1']  # Hidden layer activation
    A2 = activations['A2']  # Output layer activation (predicted probabilities)

    # Retrieve weights from the parameter dictionary
    W2 = parametres['W2']  # Weights from hidden to output layer

    # Number of samples
    m = Y.shape[1]

    # -------- Backpropagation for Output Layer --------
    # Compute the gradient of the loss with respect to Z2
    # A2 - Y is the derivative of the binary cross-entropy loss w.r.t. A2
    dz2 = A2 - Y  # Shape: (n_outputs, m)

    # Compute gradients of weights and bias for the output layer
    dW2 = (1 / m) * dz2.dot(A1.T)  # Shape: (n_outputs, n_hidden)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)  # Shape: (n_outputs, 1)

    # -------- Backpropagation for Hidden Layer --------
    # Backpropagate the error to the hidden layer
    # Derivative of sigmoid: A1 * (1 - A1)
    dz1 = W2.T.dot(dz2) * A1 * (1 - A1)  # Shape: (n_hidden, m)

    # Compute gradients of weights and bias for the hidden layer
    dW1 = (1 / m) * dz1.dot(X.T)  # Shape: (n_hidden, n_inputs)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)  # Shape: (n_hidden, 1)

    # Store all gradients in a dictionary for use in the update step
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    # Return the gradients
    return gradients
#____________________________________________________________________________
#
def update(gradients, parametres, learning_rate):
    """
    Update the weights and biases of a 2-layer neural network using gradient descent.

    Parameters:
    -----------
    gradients : dict
        Dictionary containing the gradients of the loss with respect to parameters:
            - 'dW1': gradient of loss w.r.t. W1
            - 'db1': gradient of loss w.r.t. b1
            - 'dW2': gradient of loss w.r.t. W2
            - 'db2': gradient of loss w.r.t. b2

    parametres : dict
        Dictionary containing the current weights and biases:
            - 'W1', 'b1', 'W2', 'b2'

    learning_rate : float
        The step size used to update the parameters in the direction of the negative gradient.

    Returns:
    --------
    parametres : dict
        Updated parameters after applying one step of gradient descent.
    """
    # Extract current parameters
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    # Extract computed gradients
    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    # -------- Gradient Descent Update --------
    # Subtract a fraction (learning_rate) of the gradient from each parameter
    W1 = W1 - learning_rate * dW1  # Update weights of layer 1
    b1 = b1 - learning_rate * db1  # Update biases of layer 1
    W2 = W2 - learning_rate * dW2  # Update weights of layer 2
    b2 = b2 - learning_rate * db2  # Update biases of layer 2

    # Store updated parameters in a dictionary
    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    # Return the updated parameters for use in the next iteration
    return parametres
#____________________________________________________________________________
#
def predict(X, parametres):
    """
    Make predictions using a trained 2-layer neural network.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (n_features, n_samples).
    parametres : dict
        Trained parameters of the network (weights and biases).

    Returns:
    --------
    predictions : numpy.ndarray (boolean)
        Binary predictions for each input sample.
        Values are True (1) if the output probability ≥ 0.5, False (0) otherwise.
        Shape: (1, n_samples)
    """
    # Perform forward propagation to compute the network output
    activations = forward_propagation(X, parametres)

    # Extract the final output layer activation (predicted probabilities)
    A2 = activations['A2']  # Shape: (1, n_samples)

    # Apply threshold at 0.5 to convert probabilities to binary predictions
    predictions = A2 >= 0.5  # Returns a boolean array

    # Return binary predictions (True/False)
    return predictions
#____________________________________________________________________________
#
def neural_network(X_train, y_train, n1, learning_rate=0.1, n_iter=1000):
    """
    Train a simple 2-layer neural network using gradient descent.

    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input data of shape (n_features, n_samples).
    y_train : numpy.ndarray
        Ground truth labels of shape (n_outputs, n_samples).
    n1 : int
        Number of neurons in the hidden layer.
    learning_rate : float
        Step size for gradient descent updates (default: 0.1).
    n_iter : int
        Number of training iterations (default: 1000).

    Returns:
    --------
    parametres : dict
        Trained weights and biases of the network.
    """
    # Get input and output dimensions
    n0 = X_train.shape[0]  # Number of input features
    n2 = y_train.shape[0]  # Number of output neurons (usually 1 for binary classification)

    # Initialize network parameters (weights and biases)
    parametres = initialisation(n0, n1, n2)

    # Lists to store training loss and accuracy for plotting
    train_loss = []
    train_acc = []

    # -------- Training Loop --------
    for i in range(n_iter):
        # Forward pass: compute activations
        activations = forward_propagation(X_train, parametres)

        # Backward pass: compute gradients
        gradients = back_propagation(X_train, y_train, activations, parametres)

        # Update parameters using gradients
        parametres = update(gradients, parametres, learning_rate)

        # Log performance every 10 iterations
        if i % 10 == 0:
            # Compute and store loss (log loss for classification)
            loss = log_loss(y_train.flatten(), activations['A2'].flatten())
            train_loss.append(loss)

            # Make predictions and compute accuracy
            y_pred = predict(X_train, parametres)
            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

    # -------- Visualization --------
    plt.figure(figsize=(14, 4))

    # Plot training loss over time
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Iteration (x10)')
    plt.ylabel('Log Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot training accuracy over time
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.xlabel('Iteration (x10)')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Return the trained parameters
    return parametres
#____________________________________________________________________________
#
def plot_decision_boundary(X, y, parametres):
    """
    Plot the decision boundary of a trained neural network on a 2D dataset.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (2, n_samples), i.e., 2D features only.
    y : numpy.ndarray
        Ground truth labels of shape (1, n_samples).
    parametres : dict
        Trained parameters of the neural network.
    """
    # -------- Create a Meshgrid for Plotting --------
    # Define range for x and y axes, with a margin of 1 unit
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # Step size for the grid (smaller = finer resolution)

    # Generate a meshgrid covering the feature space
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Flatten the grid and stack it into shape (2, n_points)
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Shape: (2, n_points)

    # -------- Predict on the Grid --------
    # Use the trained model to predict labels for each grid point
    predictions = predict(grid_points, parametres)  # Boolean array

    # Reshape predictions to match the shape of the meshgrid
    predictions = predictions.reshape(xx.shape)

    # -------- Plot the Decision Boundary --------
    # Color the regions based on predicted class
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap='summer')

    # Overlay the training points
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap='summer', edgecolors='k')

    # Add title and show the plot
    plt.title("Decision Boundary")
    plt.show()
#___________________________________________________________________________________________________________
#
# Sample generation
#___________________________________________________________________________________________________________
#
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print('X dimension :', X.shape)
print('y dimension :', y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()

parametres = neural_network(X, y, n1=10, n_iter=10000, learning_rate=0.1)

# Plot the decision boundary
plot_decision_boundary(X, y, parametres)

print("It worked!")