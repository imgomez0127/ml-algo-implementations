"""Computes the Maximal Marginal Relevance using JAX for hardware acceleration.

Implements the MMR algorithm proposed in:
https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf.

This library implements everything in pure python, and assumes that the input
to the maximal_marginal_relevance function is already embedded
(e.g. TF-IDF, word2vec, universal sentence encoder).
"""

from typing import Callable

import jax
import jax.numpy as jnp



@jax.jit
def cosine_similarity(
        doc1: jnp.array,
        doc2: jnp.array,
) -> jnp.array:
    return ((doc1@doc2) /
            jnp.nanprod(
                jnp.linalg.vector_norm(jnp.array([doc1, doc2]), axis=0),
                axis=0
            ))


@jax.jit
def maximal_marginal_relevance(
    query: jnp.array,
    candidate_docs: list[jnp.array],
    seen_docs: list[jnp.array],
    smoothing: float = 0.5,
    similarity_function: Callable | None = None) -> int:
    """Gets the doc with maximal marginal relevance from @candidate_docs.

    Assuming that @query, @candidate_docs, and @seen_docs are already embedded vectors.
    This will compute the maximal marginal relevance between the query and documents in
    @candidate_documents.

    Args:
        query: Embedded input user query.
        candidate_docs: Embedded documents which are determined to be relevant
            to the query user.
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
        novelty = max(
            jnp.array([
                similarity_function(candidate_doc, seen_doc)
                for seen_doc in seen_docs
            ]).reshape(-1),
            default = 0
        )

        marginal_relevance = (
            smoothing * query_similarity -
            (1-smoothing) * novelty
        )
        doc_marginal_relevance.append(marginal_relevance)
    return jnp.argmax(jnp.array(doc_marginal_relevance))

# For these test cases we use two vectors which have an equal angle from the
# query representation (The 1 Vector).
# Returns 0
print(maximal_marginal_relevance(
    jnp.array([1, 1]),
    [jnp.array([1, 0]), jnp.array([0, 1])],
    [jnp.array([0, 1])]))
# Returns 0
print(maximal_marginal_relevance(jnp.array([1, 1]),
                                 [jnp.array([1, 0]), jnp.array([0, 1])],
                                 []))
# Returns 1
print(maximal_marginal_relevance(
    jnp.array([1, 1]),
    [jnp.array([1, 0]), jnp.array([0, 1])],
    [jnp.array([1, 0])]))
