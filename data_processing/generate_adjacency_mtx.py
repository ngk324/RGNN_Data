from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./cores.txt') as f:
    cores = [line.rstrip() for line in f]
f.close()

cores = cores[:100]

rawpath = '../weights_zip/raw/'
if not os.path.exists(rawpath):
    os.makedirs(rawpath)

f = open(rawpath + 'weights_zip_A.txt', 'w')
num_previous_nodes = 0
for i in range(1,101):
    dir = '../weights_zip/weights/' + str(i)
    core = int(cores[i-1])
    one_hop_nbrs = set([core])
    one_hop_nbrs.update(find_1hop_neighbors(dir, core))
    two_hop_nbrs = find_2hop_neighbors(dir, core)
    two_hop_nbrs.update(one_hop_nbrs)
    two_hop_edges, _ = count_edges_in_subgraph(dir, two_hop_nbrs)
    two_hop_edges = list(two_hop_edges)
    two_hop_edges = sorted(two_hop_edges, key=lambda e: (e[0], e[1]))
    nodemap = {}
    nodes_sorted = list(two_hop_nbrs)
    nodes_sorted.sort()
    for j in range(len(nodes_sorted)):
        nodemap[nodes_sorted[j]] = j
    for edge in two_hop_edges:
        edge_start = nodemap[edge[0]] + num_previous_nodes
        edge_end = nodemap[edge[1]] + num_previous_nodes
        f.write(str(edge_start)+', '+str(edge_end)+'\n')
        f.write(str(edge_end)+', '+str(edge_start)+'\n')
    num_previous_nodes += len(two_hop_nbrs)

f.close()
