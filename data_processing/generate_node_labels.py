"""
TODO: Label each node with the edge weights around them
"""

from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./cores.txt') as f:
    cores = [line.rstrip() for line in f]
f.close()

num_previous_nodes = 0
pos = []
for i in range(1,351):
    
    print(f"i = {i}")
    
    dir = '../weights/weights/' + str(i)
    core = int(cores[i-1])
    one_hop_nbrs = find_1hop_neighbors(dir, core)
    two_hop_nbrs = find_2hop_neighbors(dir, core)
    all_nodes = set([core]).union(one_hop_nbrs).union(two_hop_nbrs)
    all_nodes = list(all_nodes)
    all_nodes.sort()
    
    # Map all nodes to new indices
    nodemap = {}
    for j in range(len(all_nodes)):
        nodemap[all_nodes[j]] = j

    # Generate list of weights for every node's edges
    for j in range(len(all_nodes)):
        node = all_nodes[j]
        weights = []
        nbhs = get_neighbors(dir, all_nodes, node)
        for node_ in nbhs:
            if node_ > node:
                with open(dir + '/' + str(i) + '_' + str(node) + '-' + str(node_) + '.txt') as g:
                    vals = [line.rstrip() for line in g]
                    weights.append(vals[1])
            else:
                with open(dir + '/' + str(i) + '_' + str(node_) + '-' + str(node) + '.txt') as g:
                    vals = [line.rstrip() for line in g]   
                    weights.append(vals[1])
        g.close()
        while len(weights) < 81:
            # weights.append(-1)
            weights.append(0)
        pos.append((node + num_previous_nodes, weights))
        
    # for j in range(len(all_nodes)):
    #     nodemap[all_nodes[j]] = j
    # pos.append((core + num_previous_nodes, 0))
    # for node in one_hop_nbrs:
    #     pos.append((nodemap[node] + num_previous_nodes, 1))
    # for node in two_hop_nbrs:
    #     pos.append((nodemap[node] + num_previous_nodes, 2))
    num_previous_nodes += len(all_nodes)

pos = sorted(pos, key=lambda e:e[0])

f = open('../weights/raw/weights_node_labels.txt', 'w')
for tup in pos:
    print(tup[0])
    # f.write(str(tup[1])+'\n')
    lst = tup[1]
    for i in range(80):
        f.write(str(lst[i]) + ', ')
    f.write(str(lst[-1]) + '\n')
f.close()