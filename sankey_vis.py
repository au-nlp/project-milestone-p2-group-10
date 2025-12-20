import plotly.graph_objects as go


def sankey(G, Weight):

    # 1. SETUP: Get Unique Nodes and Map to Integers
    # ---------------------------------------------------------
    # Get all unique nodes from your graph
    all_nodes = list(G.nodes())

    # Create a map: {"Intro": 0, "AI Ethics": 1, ...}
    node_to_index = {name: i for i, name in enumerate(all_nodes)}

    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]

    def hex_to_rgba(hex_code, opacity):
        # Remove the '#' if present
        hex_code = hex_code.lstrip('#')
        
        # Convert HEX to RGB integers
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        
        # Return the RGBA string
        return f"rgba({r}, {g}, {b}, {opacity})"

    # Assign a color to each node (cycling through the palette)
    node_colors = [color_palette[i % len(color_palette)] for i in range(len(all_nodes))]

    # 2. PREPARE LISTS FOR PLOTLY
    # ---------------------------------------------------------
    source_indices = []
    target_indices = []
    link_colors = []
    values = []
    labels = all_nodes  # This list maps 0 -> Label, 1 -> Label...

    # Iterate through your graph edges
    for source, target, data in G.edges(data=True):

        if source == target:
            continue

        weight = data['weight']
        
        # FILTER (Optional but Recommended):
        # If the graph is too messy, uncomment the line below to show only
        # transitions that happened more than once.
        source_hex = node_colors[node_to_index[source]]
        if weight > Weight: 
            # Get the solid color from your palette
            source_idx = node_to_index[source]
            source_hex = node_colors[source_idx]
        
            # --- [NEW] Convert to RGBA with 0.4 opacity ---
            # 0.1 = Very transparent, 1.0 = Solid
            source_rgba = hex_to_rgba(source_hex, opacity=0.4) 
            
            source_indices.append(node_to_index[source])
            target_indices.append(node_to_index[target])
            values.append(weight)
            
            # Append the TRANSPARENT color to the links
            link_colors.append(source_rgba)

    # 3. GENERATE THE SANKEY DIAGRAM
    # ---------------------------------------------------------
    fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        valuesuffix = "TWh",
        # Define the Nodes (The vertical blocks)
        arrangement = "snap",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = node_colors,  # You can replace this with a list of hex colors if you want specific colors
            align = "left"
        ),
        # Define the Links (The flowing streams)
        link = dict(
            source = source_indices, # The origin node index
            target = target_indices, # The destination node index
            value = values,          # The width of the stream (weight)
            color = link_colors
        )
    )])

    # 4. CUSTOMIZE AND SHOW
    # ---------------------------------------------------------
    fig.update_layout(
        title_text="Podcast Topic Flow Structure", 
        font_size=12,
        width= 1400,   # Width in pixels (Default is usually ~700)
        height= 550    # Height in pixels (Default is usually ~450)
        )
    fig.show()