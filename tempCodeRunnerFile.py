 # def floyd_warshall(self, start, goal):
    #     # Initialize the distance matrix with infinity for all pairs of nodes
    #     distance = list(
    #         map(lambda i: list(map(lambda j: j, i)), self.adjacency_matrix()))

    #     # Adding vertices individually
    #     for k in range(len(self.nodes)):
    #         for i in range(len(self.nodes)):
    #             for j in range(len(self.nodes)):
    #                 distance[i][j] = min(
    #                     distance[i][j], distance[i][k] + distance[k][j])

    #     # Print the adjacency matrix
    #     print('Floyd Warshall Algorithm:')
    #     print('Distance matrix every node:')
    #     for row in distance:
    #         print(f'\t{row}')

    #     return distance