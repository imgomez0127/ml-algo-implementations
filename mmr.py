"""Computes the Maximal Marginal Relevance for documents and queries.

Implements the MMR algorithm proposed in:
https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf.

This library implements everything in pure python, and assumes that the input
to the maximal_marginal_relevance function is already embedded
(e.g. TF-IDF, word2vec, universal sentence encoder).
"""
import math
from typing import Callable

def argmax(lst: list[int | float]) -> int:
    """Returns the index that has maximum value."""
    return max(enumerate(lst), key=lambda x: x[1])[0]


def l2_norm(lst: list[int | float]) -> float:
    """Computes the L2 norm (Euclidean distance) of a vector."""
    return math.sqrt(sum([x**2 for x in lst]))


def cosine_similarity(
    lst1: list[int | float],
     lst2: list[int | float]) -> float:
    """Computes the cosine similarity score between two vectors."""
    if len(lst1) != len(lst2):
        raise ValueError('Input lists must be of equal size.')
    dot_product = sum([x1 * x2 for x1, x2 in zip(lst1, lst2)])
    return dot_product/(l2_norm(lst1) * l2_norm(lst2))


def maximal_marginal_relevance(
    query: list[int | float],
    candidate_docs: list[list[int | float]], 
    seen_docs: list[list[int | float]],
    smoothing: float = 0.5,
    similarity_function: Callable | None = None) -> int:
    """Gets the doc with maximal marginal relevance from @candidate_docs.
    
    Assuming that @query, @candidate_docs, and @seen_docs are already embedded vectors.
    This will compute the maximal marginal relevance between the query and documents in
    @candidate_documents.

    Args:
        query: Embedded input user query.
        candidate_docs: Embedded documents which are determined to be relevant
            to the user query.
        seen_docs: Embedded documents which have been evaluated to have maximal
            marginal relevance.
        smoothing: Smoothing parameter to weight MMR towards being close to the
            document or being a novel from @seen_docs.
        similarity_function: Function used to compute similarity between the
            query, the candidate documents, and the seen documents.

    Returns:
        The index of the document with maximal marginal relevance.
        If @candidate_docs is empty return -1.
    """
    if not candidate_docs:
        return -1

    if similarity_function == None:
        similarity_function = cosine_similarity

    doc_marginal_relevance = []
    for candidate_doc in candidate_docs:
        query_similarity = similarity_function(query, candidate_doc)
        novelty = max([
            similarity_function(candidate_doc, seen_doc)
            for seen_doc in seen_docs
            ] + [0]
        )
        marginal_relevance = (
            smoothing * query_similarity -
            (1-smoothing) * novelty
        )
        doc_marginal_relevance.append(marginal_relevance)
    return argmax(doc_marginal_relevance)


# For these test cases we use two vectors which have an equal angle from the
# query representation (The 1 Vector).
# Returns 0
print(maximal_marginal_relevance([1, 1],[[1, 0], [0, 1]], [[0, 1]]))
# Returns 0
print(maximal_marginal_relevance([1, 1],[[1, 0], [0, 1]], []))
# Returns 1
print(maximal_marginal_relevance([1, 1],[[1, 0], [0, 1]], [[1, 0]]))
