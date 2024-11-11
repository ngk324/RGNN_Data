function [n_, l_, A_, w_] = to_dual_graph(A, w)

% Given A, which consists of multiple tuples (each representing an edge),
% contruct a new graph where each node represents an edge, and there is an
% edge between two nodes iff the two edges share a node in the original 
% graph.

% INPUT:
%  A - edges of the original graph
%  w - edge weights of the original graph

% OUTPUT:
%   n_ - nodes in the dual graph
%   l_ - node labels in the dual graph
%   A_ - edges of the dual graph
%   w_ - edge weights in the dual graphx

% Step 1: sort all rows of A
Ax = sort(A, 2);
Ax = unique(Ax, 'rows');
Ax = sort(Ax, 1);

% Step 2: create nodes from the edges
n_ = 1:size(Ax, 1);
l_ = w;

% Step 3: create edges from connectivity of old edges
A_ = [];
w_ = [];
for i = 1:(numel(n_)-1)
    e1 = Ax(i, :);
    for j = (i+1):numel(n_)
        e2 = Ax(j, :);
        ediff = e1 - e2;
        if ediff(1) == 0 || ediff(2) == 0
            A_ = [A_; i, j];
            if ediff(1) == 0
                w_ = [w_; w(e1(1))];
            else
                w_ = [w_; w(e2(2))];
            end
        end
    end
end