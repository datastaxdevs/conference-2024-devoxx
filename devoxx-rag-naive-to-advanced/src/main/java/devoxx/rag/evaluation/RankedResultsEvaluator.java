package devoxx.rag.evaluation;

import java.util.List;

/**
 * Interface for evaluating ranked results.
 */
public interface RankedResultsEvaluator<T> {

    /**
     * Computes metrics to evaluate the performance of a model.
     *
     * @param queriesResults
     *      List of ranked results for each query.
     * @param cutoffRank
     *      The maximum rank to consider (e.g., 15 for MRR@15).
     * @return The metric value.
     */
    double eval(List<RankedResults<T>> queriesResults, int cutoffRank);

}
