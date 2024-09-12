import os

"""
Re-organize all data in this folder into a large matrix
"""

dir = "../weights/weights/"
graphlst = list(map(int, os.listdir(dir)))
graphlst.sort()

def parse_filename(filename):
    """
    Extract the graph identifier, source node, and target node from the filename.
    """
    base_name = os.path.splitext(filename)[0]  # Remove the .txt extension
    parts = base_name.split('_')
    if len(parts) < 2:
        return None, None, None
    graph_id = parts[0]
    edge_id = parts[1].split('-')
    try:
        source_node = int(edge_id[0])
    except:
        return None, None, None
    target_node = int(edge_id[1])
    return graph_id, source_node, target_node

def list_files_in_directory(directory):
    """
    List all files in the given directory.
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def find_1hop_neighbors(directory, node=0):
    """
    Find all 1-hop neighbors of the given node.
    """
    one_hop_neighbors = set()
    files = list_files_in_directory(directory)
    for file in files:
        _, source_node, target_node = parse_filename(file)
        if source_node is None:
            continue
        if source_node == node:
            one_hop_neighbors.add(target_node)
        elif target_node == node:
            one_hop_neighbors.add(source_node)
    return one_hop_neighbors

def find_2hop_neighbors(directory, node=0):
    """
    Find all 2-hop neighbors of the given node.
    """
    one_hop_neighbors = find_1hop_neighbors(directory, node)
    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        two_hop_neighbors.update(find_1hop_neighbors(directory, neighbor))
    
    # Remove the original node and its 1-hop neighbors to keep only 2-hop neighbors
    two_hop_neighbors.discard(node)
    two_hop_neighbors -= one_hop_neighbors
    
    return two_hop_neighbors

def count_unique_nodes(directory):
    """
    Count the number of unique nodes in the graph.
    """
    unique_nodes = set()

    files = list_files_in_directory(directory)
    for file in files:
        _, source_node, target_node = parse_filename(file)
        unique_nodes.add(source_node)
        unique_nodes.add(target_node)

    return len(unique_nodes), unique_nodes

def count_edges_in_subgraph(directory, subgraph_nodes):
    """
    Count the number of edges within the subgraph defined by subgraph_nodes.
    """
    edge_count = 0
    edges = set()
    files = list_files_in_directory(directory)
    for file in files:
        _, source_node, target_node = parse_filename(file)
        if source_node in subgraph_nodes and target_node in subgraph_nodes:
            edge_count += 1
            edges.add((source_node, target_node))
            # edges.add((target_node, source_node))
    return edges, edge_count

def find_largest_2hop_subgraph_by_edges(directory):
    """
    Find the 2-hop subgraph with the largest number of edges.
    """
    files = list_files_in_directory(directory)
    graph_nodes = set()

    # Get all unique nodes in the graph
    for file in files:
        _, source_node, target_node = parse_filename(file)
        graph_nodes.add(source_node)
        graph_nodes.add(target_node)

    max_core = 0
    max_edges_count = 0
    max_edges = set()
    max_subgraph_nodes = set()
    
    # Find the 2-hop subgraph with the most edges
    for node in graph_nodes:
        subgraph_nodes = find_2hop_neighbors(directory, node)
        edges, edge_count = count_edges_in_subgraph(directory, subgraph_nodes)
        if edge_count > max_edges_count:
            max_core = node
            max_edges_count = edge_count
            max_subgraph_nodes = subgraph_nodes
            max_edges = edges

    return max_core, max_subgraph_nodes, max_edges_count, max_edges

def find_largest_1hop_subtree_by_edges(directory):
    files = list_files_in_directory(directory)
    graph_nodes = set()

    # Get all unique nodes in the graph
    for file in files:
        _, source_node, target_node = parse_filename(file)
        graph_nodes.add(source_node)
        graph_nodes.add(target_node)
        
    max_core = 0
    max_edges_count = 0
    max_edges = set()
    # max_subgraph_nodes = set()
    
    # Find the 1-hop subtree with the most edges
    edges, _ = count_edges_in_subgraph(directory, graph_nodes)
    nbhs = {}
    for node in graph_nodes:
        nbhs[node] = set()
    
    for edge in edges:
        source = edge[0]
        target = edge[1]
        nbhs[source].add(edge)
        nbhs[target].add(edge)
    
    for node in graph_nodes:
        if len(nbhs[node]) > max_edges_count:
            max_edges_count = len(nbhs[node])
            max_edges = nbhs[node]
            max_core = node
            # max_subgraph_nodes = 

    return max_core, max_edges_count, max_edges

def get_neighbors(directory, subgraph_nodes, node=0):
    
    edges, _ = count_edges_in_subgraph(directory, subgraph_nodes)
    nbhs = set()
    
    for edge in edges:
        source = edge[0]
        target = edge[1]
        if source == node:
            nbhs.add(target)
            continue
        if target == node:
            nbhs.add(source)
            
    nbhs = list(nbhs)
    nbhs.sort()
    return nbhs

# Usage
if __name__ == "__main__":

    max_num_neighbors = 0
    max_subgraph_size = 0
    num_neighbors_in_lgst_subgraph = 0

    f = open("cores.txt", 'w')

    for i in range(1,351):
        directory = dir + str(i)
        # Find the largest 2-hop subgraph in this graph
        max_core, max_subgraph_nodes, max_edges_count, max_edges = find_largest_2hop_subgraph_by_edges(directory)
        _, max_1hop_neighbors, _ = find_largest_1hop_subtree_by_edges(directory)
        # max_1hop_neighbors = len(find_1hop_neighbors(directory, max_core))
        print(f"In graph {i}:")
        print(f"\tthe largest 2-hop subgraph has {max_edges_count} edges")
        print(f"\tthe core is {max_core}")
        print(f"\tthe core has {max_1hop_neighbors} neighbors")

        f.write(str(max_core)+'\n')

        if max_1hop_neighbors > max_num_neighbors:
            max_num_neighbors = max_1hop_neighbors
        if max_edges_count > max_subgraph_size:
            max_subgraph_size = max_edges_count
            num_neighbors_in_lgst_subgraph = max_1hop_neighbors

    f.write(f"The biggest 2-hop subgraph among all graphs has {max_subgraph_size} edges\n")
    f.write(f"The highest degree of a node among all graphs is {max_num_neighbors}")
    f.write(f"The test data should have {max_subgraph_size + max_1hop_neighbors - num_neighbors_in_lgst_subgraph} entries")
    f.close()
