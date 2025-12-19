import networkx as nx

def Gcycle_breaker(G):

    print(f"Original Edges: {G.number_of_edges()}")

    # We loop repeatedly until there are no cycles left.
    while True:
        try:
            # 1. Try to find a cycle (e.g., A -> B -> A)
            cycle = nx.find_cycle(G)
            
            # 'cycle' is a list of edges, e.g., [(A, B), (B, C), (C, A)]
            
            # 2. Find the "Weakest Link" in this cycle
            # We want to cut the edge with the lowest weight (frequency)
            # to preserve the strongest narrative flow.
            cycle_edges_with_weights = []
            for u, v in cycle:
                weight = G[u][v]['weight']
                cycle_edges_with_weights.append((u, v, weight))
            
            # Sort by weight (smallest first)
            cycle_edges_with_weights.sort(key=lambda x: x[2])
            
            # 3. Remove the weakest edge
            u_cut, v_cut, w_cut = cycle_edges_with_weights[0]
            G.remove_edge(u_cut, v_cut)
            
            # print(f"  > Breaking Loop: Removed {u_cut} -> {v_cut} (Weight: {w_cut})")

        except nx.NetworkXNoCycle:
            # If find_cycle throws this error, it means the graph is now linear (DAG)
            print("Success! No more loops. The graph is now linear.")
            break

    print(f"Final Linear Edges: {G.number_of_edges()}")
