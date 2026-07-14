"""
Maze generation and solving example using the labyrinth library (labyrinth-py).
"""

from labyrinth.maze import Maze
from labyrinth.solve import MazeSolver


def main():
    # Create a 10x10 maze with default DFS generator
    maze = Maze(10, 10)
    
    # Print the generated maze
    print("Generated Maze:")
    print(maze)
    
    # Define starting and goal positions (as coordinates)
    start_pos = (9, 0)   # Top-left corner
    goal_pos = (9, 9)  # Bottom-right corner
    
    print(f"\nStarting position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    
  
    
    # Solve the maze
    solver = MazeSolver()
    solution_path = solver.solve(maze)
    
    if solution_path:
        print("\nSolution found!")
        print(f"Path length: {len(solution_path)} cells")
        
        # Mark the path on the maze by adding cells to maze.path
        maze.path = solution_path
        
        # Print the solved maze with the path highlighted
        print("\nMaze with Solution Path:")
        print(maze)
        
        # Print the path coordinates
        path_coords = [(cell.row, cell.column) for cell in solution_path]
        print(f"\nPath coordinates: {path_coords}")
    else:
        print("\nNo solution found!")


if __name__ == "__main__":
    main()