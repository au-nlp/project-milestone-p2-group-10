from collections import Counter
import networkx as nx

def graph_modeling(topic_model, remove_outliers=True):
    """
    Builds a NetworkX graph from the BERTopic timeline.
    
    Args:
        topic_model: The trained BERTopic model.
        remove_outliers (bool): If True, removes the node corresponding to Topic -1.
    """
    
    # 1. Get the list of IDs
    raw_topic_ids = topic_model.topics_

    # 2. Create a "Lookup Dictionary" for Topic Names
    # This maps ID 0 -> "0_music_crazy", ID -1 -> "-1_Basic Sentence Structure", etc.
    topic_info = topic_model.get_topic_info()
    id_to_name_map = dict(zip(topic_info['Topic'], topic_info['Name']))

    # 3. Convert your ID list into a Name list
    named_topic_timeline = [id_to_name_map[t_id] for t_id in raw_topic_ids]

    # Create transitions (Current Topic -> Next Topic)
    transitions = list(zip(named_topic_timeline, named_topic_timeline[1:]))

    # 4. BUILD THE NETWORKX GRAPH
    # ---------------------------------------------------------
    G = nx.DiGraph()

    transition_count = Counter(transitions)
    
    # Add edges with weights based on our counts
    for (source, target), count in transition_count.items():
        G.add_edge(source, target, weight=count)

    # 5. REMOVE OUTLIER NODE (Topic -1)
    # ---------------------------------------------------------
    if remove_outliers:
        # Find the name associated with ID -1
        outlier_label = id_to_name_map.get(-1)
        
        if outlier_label and outlier_label in G:
            # remove_node automatically removes all edges connected to it
            G.remove_node(outlier_label) 
            print(f"\n[Filtering] Successfully removed outlier hub: '{outlier_label}'")
        else:
            print("\n[Filtering] No outlier topic (-1) found in the graph.")

    # 6. INSPECT THE RESULT
    # ---------------------------------------------------------
    print(f"\nNodes (Topics): {G.number_of_nodes()}")
    print(f"Edges (Transitions): {G.number_of_edges()}")

    print("\n--- Top Transitions (Edges) ---")
    # Sort edges by weight to see the most common flows
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    # Print top 10 for brevity
    for source, target, data in sorted_edges[:10]:
        print(f"{source} --> {target} : {data['weight']} times")

    # 7. (OPTIONAL) CENTRALITY CHECK
    # ---------------------------------------------------------
    if G.number_of_nodes() > 0:
        print("\n--- New Central Hubs (Degree Centrality) ---")
        degree_centrality = nx.degree_centrality(G)
        for topic, score in sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"{topic}: {score:.2f}")
    
    return G