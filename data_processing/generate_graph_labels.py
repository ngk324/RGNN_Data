from find_maximal_subgraphs import *

def pause():
    programPause = input("Press the <ENTER> key to continue...")

with open('./cores.txt') as f:
    cores = [line.rstrip() for line in f]
f.close()

f = open('../weights_zip/raw/weights_zip_graph_labels.txt', 'w')
cores = cores[:100]
for i in range(1,101):
    dir = '../weights_zip/weights/' + str(i)
    core = int(cores[i-1])
    one_hop_nbrs = find_1hop_neighbors(dir, core)
    one_hop_nbrs = list(one_hop_nbrs)
    count = 0
    print(len(one_hop_nbrs))
    for nbr in one_hop_nbrs:
        count += 1
        if nbr > core:
            with open(dir + '/' + str(i) + '_' + str(core) + '-' + str(nbr) + '.txt') as g:
                vals = [line.rstrip() for line in g]   
        else:
            with open(dir + '/' + str(i) + '_' + str(nbr) + '-' + str(core) + '.txt') as g:
                vals = [line.rstrip() for line in g]   
        g.close()
        if count < 5:
            f.write(vals[1] + ', ')
        else:
            f.write(vals[1] + '\n')     
    # Fill the trailing entries with -1
    while count < 5:
        count += 1
        if count < 5: f.write('-1, ')
        else: f.write('-1\n')
f.close()