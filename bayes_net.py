"""
   Name    : bayes_net.py
   Author  : Ian Gomez
   Date    : December 31, 2021
   Description : On hold until I can figure out how to implement
                 an algorithm to marginalize the probabilities well
   Github  : imgomez0127@github
"""
import collections
import itertools
import pandas as pd


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
        self.variable_values = variable_values

    def compute_event_probability(self, event, conditioned_variable):
        """Computes the probability of a given event.

        This function computes the probability of a given event using the
        specified bayesian network. This is done using the variable elimination
        method of marginalizing probability distributions. for the event we
        must specify a conditioned random variable which we will be making an
        inference for.
        """
        normalization_event = event.copy()
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
                                                            event,
                                                            variable)
            # Step 3 replace all factors of Fi with Ti
            for node in intermediate_factors:
                if node in self.edges[variable]:
                    intermediate_factors[node] = intermediate_factor
        return intermediate_factors

    def multiply_factors(self, factor_tables, factor_variables):
        variable_mapping = {variable: i
                            for i, variable in enumerate(factor_variables)}
        values = [self.variable_values[factor]
                  for factor in factor_variables]
        events = list(itertools.product(*values))
        table_vars = [factor_table.columns.values
                      for factor_table in factor_tables]
        new_factor_table = []
        for event in events:
            new_factor_prob = 1
            for factor_table, table_var in zip(factor_tables, table_vars):
                indices = factor_table
                for variable in factor_variables:
                    if variable in table_var:
                        new_indices = (factor_tables[variable] ==
                                       event[variable_mapping[variable]])
                        indices = indices & new_indices
                new_factor_prob *= factor_table.loc[indices]['probability']
            new_factor_table.append([*event, new_factor_prob])
        return pd.DataFrame(new_factor_table,
                            columns=[*factor_variables, 'probability'])

    def marginalize_variable(self, factor_table, event, variable):
        remaining_variables = (set(factor_table.columns.values) -
                               {variable, 'probability'})
        remaining_variables = list(remaining_variables)
        values = [event[variable]] if variable in event else self.variable_values[variable]
        marginalized_probabilities = collections.defaultdict(int)
        for value in values:
            value_rows = factor_table[variable] == value
            for _, row in factor_table.iloc[value_rows].iterrrows():
                new_event = tuple((row[variable]
                                   for variable in remaining_variables))
                marginalized_probabilities[new_event] += row['probability']
        new_table = [[*key, value]
                     for key, value in marginalized_probabilities.items()]
        return pd.DataFrame.from_dict(new_table,
                                      columns=[*remaining_variables,
                                               "probability"])
