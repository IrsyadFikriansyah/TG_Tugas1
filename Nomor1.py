# Assignment - 1
# class     : Graph Teory - D
# created by: IrsyadFikriansyah
# date      : 7 September 2023

from queue import PriorityQueue

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.heuristic = {}

    def add_node(self, node, heuristic_value=None):
        self.nodes.add(node)
        self.edges[node] = {}
        if heuristic_value is not None:
            self.heuristic[node] = heuristic_value

    def add_edge(self, node1, node2, weight=1):
        self.edges[node1][node2] = weight
        self.edges[node2][node1] = weight

    def get_weight(self, node1, node2):
        if node1 == node2:
            return 0
        return self.edges[node1][node2]

    def print_edges(self):
        for i in self.edges.keys():
            print("%11s : " % (i))
            for j in self.edges[i]:
                print("\t%s" % (j))
            print()

    def print_heuristic(self):
        for i in self.edges.keys():
            print("%11s : %s" % (i, self.heuristic[i]))

    def adjacency_matrix(self):
        # Get the list of nodes sorted by extracting the numeric part and converting it to an integer
        sorted_nodes = sorted(list(self.nodes), key=lambda node: int(node[1:]))

        # Initialize the matrix with infinity (or any large number)
        matrix_size = len(sorted_nodes)
        adjacency_matrix = [[float('inf')] * matrix_size for _ in range(matrix_size)]

        # Fill in the matrix with edge weights
        for i in range(matrix_size):
            node_i = sorted_nodes[i]
            for j in range(matrix_size):
                node_j = sorted_nodes[j]
                if node_i == node_j:
                    adjacency_matrix[i][j] = 0  # Diagonal elements are set to 0
                elif node_j in self.edges[node_i]:
                    adjacency_matrix[i][j] = self.edges[node_i][node_j]

        return adjacency_matrix

    def a_star_search(self, start, goal):
        visited = set()
        pq = PriorityQueue()  # storing expanded nodes
        g = self.get_weight(start, start)
        h = self.heuristic[start]
        f = g + h
        pq.put((f, start, [start], 0))
        expand_count = 0

        while not pq.empty():
            # f_curr    = f score
            # current   = start
            # path      = [start]
            # distance  = total distance current from start
            f_curr, current, path, distance = pq.get()

            if current == goal:
                path_str = ' -> '.join(path)
                print(f"\nA* Search: {path_str}")
                print(f"total distance: {distance}")
                print(f"number of visited node(s): {len(visited) + 1}")
                print(f"number of expand(s): {expand_count}")
                return path

            # add the current to visited
            visited.add(current)

            for neighbor in self.edges[current]:
                if neighbor not in visited:
                    expand_count += 1
                    # print (expand_count, neighbor)
                    new_path = path + [neighbor]
                    g = f_curr - self.heuristic[current] + \
                        self.get_weight(current, neighbor)
                    h = self.heuristic[neighbor]
                    f = g + h
                    temp_dist = distance + self.get_weight(current, neighbor)
                    pq.put((f, neighbor, new_path, temp_dist))

            path_str = ' -> '.join(path)
            print(f"A* path: {path_str}")

        print("A* Search: No path found.")
        return None

    def dijkstra(self, start, goal):
        distances = {node: float('inf') for node in self.nodes}
        predecessors = {node: None for node in self.nodes}
        pq = PriorityQueue()
        pq.put((0, start))

        distances[start] = 0
        expansions = 0  # Counter for node expansions
        visited_nodes = set()

        while not pq.empty():
            current_distance, current_node = pq.get()

            # Ignore if we've already processed this node with a shorter distance
            if current_distance > distances[current_node]:
                continue

            expansions += 1
            visited_nodes.add(current_node)

            for neighbor in self.edges[current_node]:
                distance = current_distance + self.get_weight(current_node, neighbor)

                # If we find a shorter path to the neighbor, update the distance and predecessor
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    pq.put((distance, neighbor))

                    # Print the current path
                    current_path = []
                    temp_node = neighbor
                    while temp_node is not None:
                        current_path.insert(0, temp_node)
                        temp_node = predecessors[temp_node]
                    
                    path_str = ' -> '.join(current_path)
                    print(f"Dijkstra path: {path_str}")


        # Construct the path from start to goal
        path = []
        current_node = goal  # Start from the goal
        while current_node is not None:
            path.insert(0, current_node)
            current_node = predecessors[current_node]

        path_str = ' -> '.join(path)
        print(f"\nDijkstra: {path_str}")
        print(f"total distance: {distances[goal]}")
        print(f"number of visited node(s): {len(visited_nodes)}")
        print(f"number of expand(s): {expansions}")
        return path
    
    def bellman_ford(self, start, goal):
        # Initialize distances and predecessors
        distances = {node: float('inf') for node in self.nodes}
        predecessors = {node: None for node in self.nodes}
        distances[start] = 0

        # Relax edges repeatedly
        for _ in range(len(self.nodes) - 1):
            for node1 in self.nodes:
                for node2, weight in self.edges[node1].items():
                    if distances[node1] + weight < distances[node2]:
                        distances[node2] = distances[node1] + weight
                        predecessors[node2] = node1

        # Check for negative weight cycles
        for node1 in self.nodes:
            for node2, weight in self.edges[node1].items():
                if distances[node1] + weight < distances[node2]:
                    raise ValueError("Graph contains a negative weight cycle")

        print('Bellman-Ford Algorithm:')
        print('Shortest from: (distance)')
        for node, distance in distances.items():
            path = []
            current_node = node
            while current_node is not None:
                path.insert(0, current_node)
                current_node = predecessors[current_node]
            print(f"\t{start} to {node}\t: ({distance})\t{path}")

        return distances, predecessors

    def floyd_warshall(self, start, goal):
        # Initialize the distance matrix with infinity for all pairs of nodes
        distance = list(map(lambda i: list(map(lambda j: j, i)), self.adjacency_matrix()))

            # Adding vertices individually
        for k in range(len(self.nodes)):
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

        # Print the adjacency matrix
        print('Floyd Warshall Algorithm:')
        print('Distance matrix every node:')
        for row in distance:
            print(f'\t{row}')
        
        return distance

def input_graph(g):
    g.add_node("v1", 17)
    g.add_node("v2", 10.8166538264)  # c = √(9^2 + 6^2)
    g.add_node("v3", 9)
    g.add_node("v4", 11.401754251)  # c = √(9^2 + 7^2)
    g.add_node("v5", 8.5440037453)  # c = √(8^2 + 3^2)
    g.add_node("v6", 8)
    g.add_node("v7", 8.94427191)  # c = √(8^2 + 4^2)
    g.add_node("v8", 8)
    g.add_node("v9", 2)
    g.add_node("v10", 4)
    g.add_node("v11", 0)

    g.add_edge("v1", "v2", 2)
    g.add_edge("v1", "v3", 8)
    g.add_edge("v1", "v4", 1)
    g.add_edge("v2", "v3", 6)
    g.add_edge("v2", "v5", 1)
    g.add_edge("v3", "v4", 7)
    g.add_edge("v3", "v5", 5)
    g.add_edge("v3", "v6", 1)
    g.add_edge("v3", "v7", 2)
    g.add_edge("v4", "v7", 9)
    g.add_edge("v5", "v6", 3)
    g.add_edge("v5", "v8", 2)
    g.add_edge("v5", "v9", 9)
    g.add_edge("v6", "v7", 4)
    g.add_edge("v6", "v9", 6)
    g.add_edge("v7", "v9", 3)
    g.add_edge("v7", "v10", 1)
    g.add_edge("v8", "v9", 7)
    g.add_edge("v8", "v11", 2)
    g.add_edge("v9", "v10", 1)
    g.add_edge("v9", "v11", 2)
    g.add_edge("v10", "v11", 4)

def seperator():
    for _ in range(50):
        print('~', end='')
    print('\n\n')

def main():
    g = Graph()
    input_graph(g)

    start = "v1"
    goal = "v11"

    g.a_star_search(start, goal)
    seperator()
    g.dijkstra(start, goal)
    seperator()
    g.bellman_ford(start, goal)
    seperator()
    g.floyd_warshall(start, goal)

if __name__ == "__main__":
    main()

