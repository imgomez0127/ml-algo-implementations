"""
   Name    : bayes_net.py
   Author  : Ian Gomez
   Date    : December 31, 2021
   Description : On hold until I can figure out how to implement
                 an algorithm to marginalize the probabilities well
   Github  : imgomez0127@github
"""
import collections


def top_sort(graph):
    q = collections.deque()
    parent_counts = collections.Counter()
    for node, edges in graph.items():
        parent_counts[node] = parent_counts[node]
        for edge in edges:
            parent_counts[edge] += 1
    for node, parents in parent_counts.items():
        if not parents:
            q.append(node)
    ordering = []
    while q:
        cur = q.popleft()
        for node in graph[cur]:
            parent_counts[node] -= 1
            if not parent_counts[node]:
                q.append(node)
        ordering.append(cur)
    return ordering


class BayesianNetwork:
    """Represents a Bayesian Network for computing conditional probabilities.

    Takes in a dictionary of tables which maps labels to probability tables,
    and an adjacency list of edges. Computes the probability of an
    observed instance or a marginal probability of an observed instance.
    For this implementation we use pandas dataframes as tables for column
    wise indexing with strings
    """

    def __init__(self, tables, edges):
        self.tables = tables
        self.edges = edges

    def compute_probability(self, event):
        """Computes the probability of a given event.

        This function computes the probability of a given event using the
        specified bayesian network. This is done using the variable elimination
        method of marginalizing probability distributions. for the event we
        must specify a conditioned random variable which we will be making an
        inference for.
        """
        ordering = top_sort(self.edges)
