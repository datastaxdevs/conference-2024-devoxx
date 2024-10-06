package devoxx.rag.evaluation.ndcg;

import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.RankedResultsEvaluator;

import java.util.*;

/**
 * Evaluator for computing Normalized Discounted Cumulative Gain (NDCG).
 *
 * @param <T>
 *      The type of the items in the ranked results.
 */
public class NormalizedDiscountedCumulativeGain<T> implements RankedResultsEvaluator<T> {

    /**
     * Computes the NDCG@K for a list of queries.
     *
     * @param queriesResults
     *      List of ranked results for each query.
     * @param cutoffRank
     *      The maximum rank to consider (e.g., 10 for NDCG@10).
     * @return The NDCG value.
     */
    @Override
    public double eval(List<RankedResults<T>> queriesResults, int cutoffRank) {
        double sumNDCG = 0.0;
        int validQueryCount = 0;

        for (RankedResults<T> rankedResult : queriesResults) {
            Map<T, Integer> relevanceGrades = rankedResult.getRelevanceGrades();

            if (relevanceGrades == null || relevanceGrades.isEmpty()) {
                // No relevance grades for this query, skip it
                continue;
            }

            validQueryCount++;

            // Compute DCG@K
            double dcg = computeDCG(rankedResult.getResults(), relevanceGrades, cutoffRank);

            // Compute IDCG@K (ideal DCG)
            double idcg = computeIDCG(relevanceGrades, cutoffRank);

            double ndcg = idcg > 0.0 ? dcg / idcg : 0.0;
            sumNDCG += ndcg;
        }

        // Avoid division by zero if validQueryCount is zero
        return validQueryCount > 0 ? sumNDCG / validQueryCount : 0.0;
    }

    private double computeDCG(NavigableMap<Double, List<T>> results, Map<T, Integer> relevanceGrades, int cutoffRank) {
        double dcg = 0.0;
        int rank = 1;

        for (Map.Entry<Double, List<T>> entry : results.entrySet()) {
            List<T> docsAtScore = entry.getValue();

            for (T doc : docsAtScore) {
                if (rank > cutoffRank) {
                    break;
                }
                int grade = relevanceGrades.getOrDefault(doc, 0); // Default to 0 if not in relevance grades
                dcg += (Math.pow(2, grade) - 1) / (Math.log(rank + 1) / Math.log(2));
                rank++;
            }

            if (rank > cutoffRank) {
                break;
            }
        }

        return dcg;
    }

    private double computeIDCG(Map<T, Integer> relevanceGrades, int cutoffRank) {
        // Sort the relevance grades in descending order
        List<Integer> sortedGrades = new ArrayList<>(relevanceGrades.values());
        sortedGrades.sort(Collections.reverseOrder());

        double idcg = 0.0;
        int rank = 1;

        for (int grade : sortedGrades) {
            if (rank > cutoffRank) {
                break;
            }
            idcg += (Math.pow(2, grade) - 1) / (Math.log(rank + 1) / Math.log(2));
            rank++;
        }

        return idcg;
    }
}
