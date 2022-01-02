"""
   Name    : bayes_net.py
   Author  : Ian Gomez
   Date    : December 31, 2021
   Description : On hold until I can figure out how to implement
                 an algorithm to marginalize the probabilities well
   Github  : imgomez0127@github
"""
import heapq
import collections
import itertools
import pandas as pd


def top_sort(graph):
    """Performs topological sort

    Uses topological sort to get the ordering for computing marginalized
    probability distributions for the Variable Elimination algorithm.
    This will output the ordering sorted by the lower amount of outgoing
    edges for a more optimal VE ordering.
    """
    q = []
    parent_counts = collections.Counter()
    for node, edges in graph.items():
        parent_counts[node] = parent_counts[node]
        for edge in edges:
            parent_counts[edge] += 1
    for node, parents in parent_counts.items():
        if not parents:
            heapq.heappush(q, (len(graph[node]), node))
    ordering = []
    while q:
        _, cur = heapq.heappop(q)
        for node in graph[cur]:
            parent_counts[node] -= 1
            if not parent_counts[node]:
                heapq.heappush(q, (len(graph[node]), node))
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
        print(ordering)
        # Repeat for bayesian normalization constant
        conditioned_probability = self.eliminate_variables(event, ordering)
        normalization_probability = 0
        for value in self.variable_values[conditioned_variable]:
            normalization_event[conditioned_variable] = value
            normalization_factor = self.eliminate_variables(
                normalization_event, ordering)
            normalization_probability += normalization_factor
        return conditioned_probability/normalization_probability

    def eliminate_variables(self, event, ordering):
        """ Computes the VE algorithm.

        This functions computes the VE algorithm specified in Stanford's
        CS228 lecture notes.
        """
        intermediate_factors = self.tables.copy()
        for variable in ordering[:-1]:
            # Step 1 of VE algorithm multiply all factors containing Xi
            table_ids = set()
            tables = []
            possible_factors = self.edges[variable]+[variable]
            for node in possible_factors:
                if id(intermediate_factors[node]) not in table_ids:
                    tables.append(intermediate_factors[node])
                    table_ids.add(id(intermediate_factors[node]))
            factor_variables = self.edges[variable]+[variable]
            factor_table = self.multiply_factors(tables)
            # Step 2 marginalize to get new factor
            intermediate_factor = self.marginalize_variable(factor_table,
                                                            event,
                                                            variable)
            # Step 3 replace all factors of Fi with Ti
            for node in factor_variables:
                if node in self.edges[variable]:
                    intermediate_factors[node] = intermediate_factor
        final_table = intermediate_factors[ordering[-1]]
        print(final_table)
        for variable in final_table.columns.values:
            if variable not in (ordering[-1], 'probability'):
                final_table = self.marginalize_variable(final_table, event, variable)
        if ordering[-1] in event:
            event_row = final_table.loc[final_table[ordering[-1]] == event[ordering[-1]]]
            return float(event_row['probability'].sum())
        return float(intermediate_factors['probability'].sum())

    def multiply_factors(self, factor_tables):
        """Performs the multiply factors step of the VE algorithm.

        This function computes the multiplicaton of factors.
        This essentially computes a big table with where all
        possible events are observed for all factor variables that are input.
        It then computes the probability of the factor table.
        """
        variables = list({header
                          for factor_table in factor_tables
                          for header in factor_table.columns.values
                          if header != 'probability'})

        variable_mapping = {variable: i for i, variable in enumerate(variables)}
        values = [self.variable_values[factor] for factor in variables]
        events = list(itertools.product(*values))
        table_vars = [list(factor_table.columns.values)
                      for factor_table in factor_tables]
        new_factor_table = []

        for event in events:
            new_factor_prob = 1
            for factor_table, table_var in zip(factor_tables, table_vars):
                indices = factor_table['probability'] >= 0.0
                for variable in variables:
                    if variable in table_var:
                        new_indices = (factor_table[variable] ==
                                       event[variable_mapping[variable]])
                        indices = indices & new_indices
                new_factor_prob *= float(factor_table.loc[indices]['probability'])
            new_factor_table.append([*event, new_factor_prob])
        return pd.DataFrame(new_factor_table,
                            columns=[*variables, 'probability'])

    def marginalize_variable(self, factor_table, event, variable):
        """Performs marginalization step of VE algorithm.

        This function performs the marginalization step of the VE algorithm.
        It marginalizes the factor table based on the given event. Where we
        marginalize the probability of P(var) if var is not observed
        in the event. Otherwise we marginalize P(var=x) if x is given in the
        event.
        """
        remaining_variables = (set(factor_table.columns.values) -
                               {variable, 'probability'})

        remaining_variables = list(remaining_variables)
        values = [event[variable]] if variable in event else self.variable_values[variable]
        marginalized_probabilities = collections.defaultdict(int)

        for value in values:
            value_rows = factor_table[variable] == value
            for _, row in factor_table.loc[value_rows].iterrows():
                new_event = tuple((row[variable]
                                   for variable in remaining_variables))
                marginalized_probabilities[new_event] += row['probability']

        new_table = [[*key, value]
                     for key, value in marginalized_probabilities.items()]
        return pd.DataFrame(new_table,
                            columns=[*remaining_variables, 'probability'])


def main():
    rain_table = pd.DataFrame([[True, 0.2],
                               [False, 0.8]],
                              columns=['rain', 'probability'])
    sprinkler_table = pd.DataFrame([[True, 0.1],
                                    [False, 0.9]],
                                   columns=['sprinkler', 'probability'])
    grass_table = pd.DataFrame([[True, False, True, 1.0],
                                [True, True, True, 1.0],
                                [True, True, False, 0.9],
                                [True, False, False, 0.0],
                                [False, False, True, 0.0],
                                [False, True, True, 0.0],
                                [False, True, False, 0.1],
                                [False, False, False, 1.0]],
                               columns=['grass wet', 'sprinkler',
                                        'rain', 'probability'])
    jack_table = pd.DataFrame([[True, True, 1.0],
                               [True, False, 0.2],
                               [False, True, 0.0],
                               [False, False, 0.8]],
                              columns=['jack', 'rain', 'probability'])
    tables = {
        'rain': rain_table,
        'sprinkler': sprinkler_table,
        'grass wet': grass_table,
        'jack': jack_table
    }
    edges = {
        'rain': ['grass wet', 'jack'],
        'sprinkler': ['grass wet'],
        'grass wet': [],
        'jack': []
    }
    variable_values = {
        'rain': [True, False],
        'sprinkler': [True, False],
        'grass wet': [True, False],
        'jack': [True, False]
    }
    network = BayesianNetwork(tables, edges, variable_values)
    event = {'sprinkler': True, 'grass wet': True, 'jack': True}
    print(f'Probability of {event}')
    print(network.compute_event_probability(event, 'sprinkler'))


if __name__ == '__main__':
    main()
