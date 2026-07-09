import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_circles
from sklearn.metrics import log_loss, accuracy_score

def initialization(dimensions):
    
    parameters = {}
    C = len(dimensions)
    
    for c in range(1, C):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c-1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)
        
    return parameters  

def forward_propagation(X, parameters):
    
    activations = {'A0' : X}
    
    C = len(parameters) // 2
    
    for c in range(1, C + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c-1)])+ parameters['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
        
    return activations      

def back_propagation(y, activations, parameters):
    m = y.shape[1]
    C = len(parameters) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parameters['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

    return gradients

def update (gradients, parameters, learning_rate):
    C = len(parameters) // 2
    for c in range(1, C + 1):
        parameters['W' + str(c)] -= learning_rate * gradients['dW' + str(c)]
        parameters['b' + str(c)] -= learning_rate * gradients['db' + str(c)]
    
    return parameters    

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    A_final = activations['A' + str(len(parameters) // 2)]
    return A_final >= 0.5

def neural_network(X_train, y_train, dimensions, learning_rate=0.1, n_iter=1000):
    
    parameters = initialization(dimensions)
    history = {'loss': [], 'acc': [], 'params': [], 'grid_probs': []}

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

    for i in range(n_iter):
        activations = forward_propagation(X_train, parameters)
        gradients = back_propagation(y_train, activations, parameters)
        parameters = update(gradients, parameters, learning_rate)
        '''
        if i % 100 == 0 or i == n_iter - 1:
            A_final = activations['A' + str(len(dimensions) - 1)]
            loss = log_loss(y_train.flatten(), A_final.flatten())
            acc = accuracy_score(y_train.flatten(), (A_final >= 0.5).flatten())
            history['loss'].append(loss)
            history['acc'].append(acc)
            history['params'].append(parameters.copy())
            print(f"Iteration {i}/{n_iter} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")
        '''
        if i % 100 == 0 or i == n_iter - 1:
            # 1. Salva snapshot dei parametri
            history['params'].append(parameters.copy())
        
            # 2. Calcola e salva grid_probs
            grid_activations = forward_propagation(grid_points, parameters)
            grid_probs = grid_activations['A' + str(len(parameters) // 2)].reshape(xx.shape)
            history['grid_probs'].append(grid_probs)
        
            # 3. Loss e accuracy
            A_final = activations['A' + str(len(dimensions) - 1)]
            loss = log_loss(y_train.flatten(), A_final.flatten())
            acc = accuracy_score(y_train.flatten(), (A_final >= 0.5).flatten())
            history['loss'].append(loss)
            history['acc'].append(acc)

            print(f"Iteration {i}/{n_iter} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")

    return parameters, history, xx, yy

def plot_decision_boundary(X, y, parameters, resolution=0.01):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    activations = forward_propagation(grid_points, parameters)
    A_final = activations['A' + str(len(parameters) // 2)]
    Z = A_final.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('white')
    contour = plt.contourf(xx, yy, Z, levels=50, cmap='bwr', alpha=0.2)
    plt.contour(xx, yy, Z, levels=[0.5], colors='orange', linewidths=2)
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap='coolwarm', edgecolors='k', s=50)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(contour, label='Class Probability')
    plt.show()

# -----------------------------
# Animation
# -----------------------------
def create_decision_boundary_animation(X, y, history, xx, yy):
    """
    Generate animation of decision boundary evolution for any depth network,
    using precomputed grid_probs saved during training.
    
    Args:
        X : input data (2, n)
        y : labels (1, n)
        history : dictionary containing 'grid_probs', 'loss', 'acc'
        xx, yy : meshgrid used during training
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        ax.clear()
        ax.set_facecolor('white')

        # Use saved grid probabilities
        probs = history['grid_probs'][frame]
        ax.contourf(xx, yy, probs, levels=50, cmap='bwr', alpha=0.2)
        ax.contour(xx, yy, probs, levels=[0.5], colors='orange', linewidths=2)
        ax.scatter(X[0, :], X[1, :], c=y.flatten(), cmap='coolwarm', edgecolors='k', s=50)

        ax.set_title(f"Iteration: {frame * 100}\nLoss: {history['loss'][frame]:.4f}  Accuracy: {history['acc'][frame]:.2f}")
        return []

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(history['grid_probs']),
        interval=120,
        blit=False
    )

    plt.show()

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
# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    '''
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    print(X.shape)
    print(y.shape)
    '''
    X, y = generate_composite_circles()
    print(X.shape)
    print(y.shape)
    
    dimensions = [2, 16, 32, 1]  # Architecture
    print("Training...")
    params, history, xx, yy = neural_network(X, y, dimensions, learning_rate=0.1, n_iter=20000)

    print("Plotting decision boundary...")
    plot_decision_boundary(X, y, params)

    print("Creating animation...")
    create_decision_boundary_animation(X, y, history, xx, yy)
    
    print("It worked!")