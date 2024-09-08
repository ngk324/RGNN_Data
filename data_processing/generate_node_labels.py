from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./cores.txt') as f:
    cores = [line.rstrip() for line in f]
f.close()

num_previous_nodes = 0
pos = []
for i in range(1,101):
    dir = './weights/' + str(i)
    core = int(cores[i-1])
    one_hop_nbrs = find_1hop_neighbors(dir, core)
    two_hop_nbrs = find_2hop_neighbors(dir, core)
    all_nodes = set([core]).union(one_hop_nbrs).union(two_hop_nbrs)
    all_nodes = list(all_nodes)
    all_nodes.sort()
    nodemap = {}
    for j in range(len(all_nodes)):
        nodemap[all_nodes[j]] = j
    pos.append((core + num_previous_nodes, 0))
    for node in one_hop_nbrs:
        pos.append((nodemap[node] + num_previous_nodes, 1))
    for node in two_hop_nbrs:
        pos.append((nodemap[node] + num_previous_nodes, 2))
    num_previous_nodes += len(all_nodes)

pos = sorted(pos, key=lambda e:e[0])

f = open('./raw/weights_zip_node_labels.txt', 'w')
for tup in pos:
    print(tup)
    f.write(str(tup[1])+'\n')
f.close()