import itertools
import statistics
import random 
import time
import matplotlib.pyplot as plt

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

########################################
# NAIVE
########################################
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
            return True
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

        # Build the induced subgraph H[n][n]
        H = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                H[a][b] = graph[L[a]][L[b]]

        # Find the indexes of start and end in L
        s = L.index(start)
        t = L.index(end)

        if all_permutations(H, s, t):
            return True  

    # No valid subset + Hamiltonian path found
    return False

# Part 2

########################################
# OPTIMIZED
########################################
def bfs_component(graph, start, n):
    """
    Reconstructs the hidden subgraph containing 'start' using BFS.
    Each of the 3 subgraphs has exactly n vertices.
    """
    visited = set([start])
    queue = [start]

    while queue and len(visited) < n:
        u = queue.pop(0)
        for v in range(len(graph)):
            if graph[u][v] == 1 and v not in visited:
                visited.add(v)
                queue.append(v)

    return visited

def hamiltonian_optimized(graph, start, end):
    """
    Optimized Hamiltonian path check that identifies the correct subgraph
    even after random vertex permutation.
    """
    N = len(graph)  
    n = N // 3

    # Reconstruct the correct subgraph of size n using BFS
    component = bfs_component(graph, start, n)

    # If end is not in the same BFS component, no Hamiltonian path exists
    if end not in component:
        return False
    
    # Convert BFS component to an ordered list of vertices
    L = sorted(component)
    
    # Build adjacency matrix H for this subgraph
    H = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            u, v = L[i], L[j]
            H[i][j] = graph[u][v]

    # Find indices of start and end in L
    s = L.index(start)
    t = L.index(end)

    # Check permutations inside the correct subgraph
    return all_permutations(H, s, t)

# Part 3

########################################
# BONUS
########################################
def hamiltonian_bonus(graph, start, end) -> bool:
    pass  # Placeholder for bonus implementation
    # return True if a Hamiltonian path exists from start to end, else False

def main():
    plot_time_vs_n(start_n=4, end_n=13, optimized=True, logy=True, logx=False)
 
def run_naive_stats(n):
    times = []
    avg_time = 0
    for _ in range(10):  # 10 rounds
        graph, start, end = generate_tricky_graph(n)
        start_time = time.time()
        hamiltonian_naive(graph, start, end)
        end_time = time.time()
        times.append(end_time - start_time)
    return statistics.mean(times)

def run_optimized_stats(n):
  times = []
  avg_time = 0
  for _ in range(10):  # 10 rounds
      graph, start, end = generate_tricky_graph(n)
      start_time = time.perf_counter()
      hamiltonian_optimized(graph, start, end)
      end_time = time.perf_counter()
      times.append(end_time - start_time)
  return statistics.mean(times)

def plot_time_vs_n(start_n=4, end_n=8, optimized=False, logy=True, logx=False):
    """
    Plots average running time vs n for the Hamiltonian path algorithms.
    If optimized=True, uses the optimized algorithm; else uses naive.
    Set logy = True for log scale on y-axis, logx = True for log scale on x-axis.
    """
    ns = list(range(start_n, end_n + 1))
    times = []
    for n in ns:
        if optimized:
            avg_time = run_optimized_stats(n)
        else:
            avg_time = run_naive_stats(n)
        times.append(avg_time)
        print(f"n={n}: avg_time={avg_time:.8f} seconds")
    label = "Optimized" if optimized else "Naive"
    plt.figure(figsize=(8, 5))
    plt.plot(ns, times, marker='o', label=label)
    plt.xlabel('n (subgraph size)')
    plt.ylabel('Average Time (seconds)')
    plt.title(f"Hamiltonian Path: Time vs n ({label})")
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.6)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()