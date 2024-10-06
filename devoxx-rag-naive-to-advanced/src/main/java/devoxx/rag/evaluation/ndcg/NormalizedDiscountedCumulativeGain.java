package devoxx.rag.evaluation.ndcg;

import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.RankedResultsEvaluator;
import devoxx.rag.evaluation.relevance.ExactMatchRelevanceChecker;
import devoxx.rag.evaluation.relevance.RelevanceChecker;
import lombok.Data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;

/**
 * Evaluator for computing Normalized Discounted Cumulative Gain (NDCG).
 *
 */
@Data
public class NormalizedDiscountedCumulativeGain<T> implements RankedResultsEvaluator<T> {

    private static final int DEFAULT_CUTOFF_RANK = 15;

    private final int cutOffRank;

    /** Needed to evaluate the relevance of the retrieved documents. */
    private final RelevanceChecker<T> relevanceChecker;

    public NormalizedDiscountedCumulativeGain() {
        this( new ExactMatchRelevanceChecker<T>(), DEFAULT_CUTOFF_RANK);
    }

    public NormalizedDiscountedCumulativeGain(RelevanceChecker<T> relevanceChecker, int cutOffRank) {
        this.cutOffRank = cutOffRank;
        this.relevanceChecker = relevanceChecker;
    }

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
            double dcg = computeDCG(rankedResult.getMatches(), relevanceGrades, cutoffRank);

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
