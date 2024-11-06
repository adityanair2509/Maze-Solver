import cv2
import numpy as np
import os
from collections import deque

class MazeImageSolver:
    def __init__(self, image_path):
        self.image_path = image_path
        self.maze, self.binary_maze = self.process_image()

    def process_image(self):
        if not os.path.exists(self.image_path):
            raise ValueError("Image not found or unable to read.")

        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_maze = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Scale binary maze to 0 (path) and 1 (wall)
        binary_maze = (binary_maze // 255).astype(np.uint8)

        # Display the binary maze to check if paths and walls are correctly represented
        cv2.imshow("Binary Maze", binary_maze * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image, binary_maze

    def solve_maze(self):
        start = (0, 0)
        end = (self.binary_maze.shape[0] - 1, self.binary_maze.shape[1] - 1)
        
        # Run BFS to find the shortest path
        path = self.bfs_solve(start, end)

        if path:
            self.draw_solution(path)
        else:
            print("No path found.")

    def bfs_solve(self, start, end):
        queue = deque([start])
        visited = set()
        parent = {start: None}

        while queue:
            current = queue.popleft()
            if current == end:
                return self.reconstruct_path(parent, end)

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and self.binary_maze[neighbor[0], neighbor[1]] == 0:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None

    def reconstruct_path(self, parent, end):
        path = []
        while end:
            path.append(end)
            end = parent[end]
        path.reverse()
        return path

    def get_neighbors(self, pos):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        for d in directions:
            neighbor = (pos[0] + d[0], pos[1] + d[1])
            if 0 <= neighbor[0] < self.binary_maze.shape[0] and 0 <= neighbor[1] < self.binary_maze.shape[1]:
                neighbors.append(neighbor)
        return neighbors

    def draw_solution(self, path):
        solution_image = cv2.cvtColor(self.binary_maze * 255, cv2.COLOR_GRAY2BGR)
        for pos in path:
            solution_image[pos[0], pos[1]] = (0, 0, 255)  # Draw the path in red

        cv2.imshow("Maze Solution", solution_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main execution
image_path = input("Enter the path to the maze image: ")
solver = MazeImageSolver(image_path)
solver.solve_maze()
