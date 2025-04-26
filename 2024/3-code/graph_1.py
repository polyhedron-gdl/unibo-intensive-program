import matplotlib.pyplot as plt
import numpy as np

class PointConnector:
    def __init__(self, num_points):

        test = np.array([[6.8,0],[5.5,2.8],[2.1,4.5],[-2.1,4.5],[-5.5,2.8],[-6.8,0],[-5.5,-2.8],[-2.1,-4.5],[2.1,-4.5],[5.5,-2.8]])
        self.A = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        np.savetxt("graph-matrix.csv", self.A, delimiter=';')
            
        self.num_points = len(test) #num_points
        self.points     = test # np.random.rand(num_points, 2)  # Generate random points
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(self.points[:, 0], self.points[:, 1])
        self.clicked_points = []

        # Connect event handler for click events
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is not self.ax:
            return

        # Find the closest point
        distance = np.sqrt((self.points[:, 0] - event.xdata) ** 2 + (self.points[:, 1] - event.ydata) ** 2)
        closest_point_index = np.argmin(distance)
        self.clicked_points.append(self.points[closest_point_index])

        # If two points have been selected, draw a line between them
        if len(self.clicked_points) == 2:
            P1 = np.array([self.clicked_points[0][0],self.clicked_points[0][1]])
            P2 = np.array([self.clicked_points[1][0],self.clicked_points[1][1]])
            
            # Find the indices where the rows match the variable_to_compare
            matches = np.where(np.all(self.points == P1, axis=1))[0]
            for match in matches: idx1 = match
            matches = np.where(np.all(self.points == P2, axis=1))[0]
            for match in matches: idx2 = match
            self.A[idx1][idx2] = self.A[idx2][idx1] = 1
            #print(idx1, " - ", idx2)
            #print(self.A)
            np.savetxt("graph-matrix.csv", self.A, delimiter=';')
            
            self.ax.plot([self.clicked_points[0][0], self.clicked_points[1][0]],
                         [self.clicked_points[0][1], self.clicked_points[1][1]], 'r-')
            self.fig.canvas.draw()  # Redraw the figure to show the line
            self.clicked_points = []  # Reset for the next pair of points

if __name__ == "__main__":
    #num_points = int(input("Enter the number of points to generate: "))
    pc = PointConnector(10)
    plt.show()
