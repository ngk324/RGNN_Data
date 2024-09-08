To unzip the data file, use the following command in the local directory of this repo:
```
mkdir weights_zip
unzip weights_zip.zip -d ./weights_zip
```
To generate the graph data, run the following commands:
```
cd data_processing/
python3 find_maximal_subgraphs.py    # Find the largest 2-hop subgraphs in each graph
python3 generate_adjacency_mtx.py    # Generate a sparse adjacency matrix for all graphs
python3 generate_graph_labels.py     # Label each graph with the desired edge weights around the core of its largest 2-hop subgraph
python3 generate_graph_indicators.py # Label each node with the graph that it belongs to
python3 generate_node_labels.py      # Label each node in the subgraph with its position (0->core, 1->one-hop, 2->two-hop)
python3 generate_edge_labels.py      # Label each edge in the subgraph with its weight after one step of global optimization
```
After preparing the raw data, one may proceed to training the model:
```
cd ../experiment/

```

