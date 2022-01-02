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
    """Performs topological sort

    Uses topological sort to get the ordering for computing marginalized
    probability distributions for the Variable Elimination algorithm.
    """
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

    def __init__(self, tables, edges, variable_values):
        self.tables = tables
        self.edges = edges
        self.variable_values

    def compute_probability(self, event, conditioned_variable):
        """Computes the probability of a given event.

        This function computes the probability of a given event using the
        specified bayesian network. This is done using the variable elimination
        method of marginalizing probability distributions. for the event we
        must specify a conditioned random variable which we will be making an
        inference for.
        """
        normalization_event = event.copy
        ordering = top_sort(self.edges)
        # Repeat for bayesian normalization constant
        event_factors = self.eliminate_variables(event, ordering)
        conditioned_probability = event_factors[conditioned_variable]
        normalization_probability = 0
        for value in self.variable_values[conditioned_variable]:
            normalization_event[conditioned_variable] = value
            normalization_factors = self.eliminate_variables(
                normalization_event, ordering)
            normalization_probability += normalization_factors[
                conditioned_variable]
        return conditioned_probability/normalization_probability

    def eliminate_variables(self, event, ordering):
        intermediate_factors = self.tables.copy()
        for variable in ordering:
            # Step 1 of EV algorithm multiply all factors containing Xi
            factor_variables = [node for node in intermediate_factors.items()
                                if node in self.edges[variable]]
            factors = [table for node, table in intermediate_factors.items()
                       if node in self.edges[variable]]
            factor_table = self.multiply_factors(factors, factor_variables)
            # Step 2 marginalize to get new factor
            intermediate_factor = self.marginalize_variable(factor_table,
                                                            event)
            # Step 3 replace all factors of Fi with Ti
            for node in intermediate_factors:
                if node in self.edges[variable]:
                    intermediate_factors[node] = intermediate_factor
        return intermediate_factors

    def multiply_factors(self, factors, factor_variables):
        for factor in factor_variables:
            pass

    def marginalize_variable(self, factors, event):
        pass
