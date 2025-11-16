import itertools
import random 
# ------------------------------
# Graph construction (spec §2.1)
# ------------------------------
def add_connected_subgraph(graph, vertices):
    L = vertices[:]
    random.shuffle(L)  # Step 1: random spanning path to ensure connectivity
    for k in range(len(L) - 1):
        u, v = L[k], L[k + 1]
        graph[u][v] = graph[v][u] = 1
    # Add extra random edges with p=1/2 among remaining pairs
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            if graph[u][v] == 0 and random.random() < 0.5:
                graph[u][v] = graph[v][u] = 1

def generate_tricky_graph(n):
    N = 3 * n
    start = random.randint(0, N - 1)
    while True:
        end = random.randint(0, N - 1)
        if end != start:
            break

    graph = [[0] * N for _ in range(N)]

    A = list(range(0, n))
    B = list(range(n, 2 * n))
    C = list(range(2 * n, 3 * n))

    add_connected_subgraph(graph, A)
    add_connected_subgraph(graph, B)
    add_connected_subgraph(graph, C)

    # Hide structure by permuting vertex labels; remap start/end accordingly
    # perm is new_to_old, new_to_old[0] means the prev version of the new vertex 0
    # inv is old_to_new, old_to_new[6] means the new version of the old vertex 6
    perm = list(range(N))
    random.shuffle(perm)
    inv = [0] * N
    for i, p in enumerate(perm):
        inv[p] = i
    
    permuted = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            permuted[i][j] = graph[perm[i]][perm[j]]
    graph = permuted
    start = inv[start]
    end = inv[end]

    return graph, start, end

# Part 1
def hamiltonian_check(H, perm) -> bool:
    """
    Hamiltonian Check is a helper function to check if a given graph `H` has
    necessary edges given in the `perm` permutation of vertices.
    
    Parameters:
        H: Graph represented by an 2D adjacency matrix (list of lists)
        perm: A list of vertex indices representing a permutation of vertices from 0 to n-1
        (list of integers)
    Returns:
        bool: True if `perm` represents a Hamiltonian path, False otherwise
    """
    n = len(H)
    for i in range(n - 1):
        if H[perm[i]][perm[i + 1]] == 0:
            return False
    return True

def all_permutations(H, s, t) -> bool:
    """
    All Permutations is an algorithm that checks all permutations of vertices
    in a graph H to determine if there exists a Hamiltonian path from vertex s to vertex t.
    
    Parameters:
        H: adjacency matrix of the graph (list of lists)
        s: start vertex (int)
        t: end vertex (int)
    Returns:
        bool: True if a Hamiltonian path exists from s to t, else False
    """
    n = len(H)

    # Vertices except start and end
    middle_vertices = [v for v in range(n) if v != s and v != t]
    
    for middle_perm in itertools.permutations(middle_vertices):
        perm = [s] + list(middle_perm) + [t] # add start and end vertices
        if hamiltonian_check(H, perm):
            return True # Found a valid Hamiltonian path
    return False # None of the permutations formed a Hamiltonian path

def hamiltonian_naive(graph, start, end) -> bool:
    """
    Naive Hamiltonian Path algorithm that checks all subsets of vertices of size n
    and all permutations of those vertices to determine if there exists a Hamiltonian path
    
    Parameters:
        graph: adjacency matrix of the graph of the form 3n x 3n (list of lists)
        start: start vertex (int)
        end: end vertex (int)
    Returns:
        bool: True if a Hamiltonian path exists from start to end, else False
    """
    N = len(graph)       # N = 3n
    n = N // 3
    V = set(range(N))    # V = {0, 1, ..., 3n−1}

    # Check all subsets S of V with size n
    for S in itertools.combinations(V, n):
        if start not in S or end not in S: # if not both in S, skip
            continue

        L = sorted(list(S))

        # Step 4: Build the induced subgraph H[n][n]
        H = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                H[a][b] = graph[L[a]][L[b]]

        # Find the indexes of start and end in L
        s = L.index(start)
        t = L.index(end)

        if all_permutations(H, s, t):
            return True  # Found a valid subset + Hamiltonian path

    # No valid subset + Hamiltonian path found
    return False

# Part 2
def hamiltonian_optimized(graph, start, end) -> bool:
    pass # Placeholder for optimized implementation
    # return True if a Hamiltonian path exists from start to end, else False

# Bonus Part
def hamiltonian_bonus(graph, start, end) -> bool:
    pass  # Placeholder for bonus implementation
    # return True if a Hamiltonian path exists from start to end, else False

def main():
 
    n = 3 # You can change n to generate larger or smaller graphs
    graph, start, end = generate_tricky_graph(n)
   
    """
    n_values = [4, 5, 6, 7, 8] 
    experimental_analysis(n_values)
    """

# Helper function for obtaining experimental results
def experimental_analysis(n_values, repeat_count=10):
    import time
    import matplotlib.pyplot as plt
    print("Starting experimental analysis...")
    
    avg_durations = []

    for n in n_values:
        durations_for_n = []

        for _ in range(repeat_count):

            graph, start, end = generate_tricky_graph(n)

            # Measure only the naive algorithm execution time
            start_time = time.perf_counter()
            hamiltonian_naive(graph, start, end)
            end_time = time.perf_counter()

            durations_for_n.append(end_time - start_time)

        avg_time_for_n = sum(durations_for_n) / repeat_count
        avg_durations.append(avg_time_for_n)
        print(f"n={n}, avg time over {repeat_count} runs: {avg_time_for_n} sec")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, avg_durations, marker='o')
    plt.title("Naive Hamiltonian Path Execution Time")
    plt.xlabel("n (subset size)")
    plt.ylabel("Average execution time (seconds)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()