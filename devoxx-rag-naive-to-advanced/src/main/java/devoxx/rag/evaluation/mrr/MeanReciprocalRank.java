package devoxx.rag.evaluation.mrr;

import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.RankedResultsEvaluator;
import devoxx.rag.evaluation.relevance.ExactMatchRelevanceChecker;
import devoxx.rag.evaluation.relevance.RelevanceChecker;
import lombok.Data;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;

/**
 * Proposition for an implementation of MRR
 */
@Data
public class MeanReciprocalRank<T> implements RankedResultsEvaluator<T> {

    private static final int DEFAULT_CUTOFF_RANK = 15;

    private final int cutOffRank;

    /** Needed to evaluate the relevance of the retrieved documents. */
    private final RelevanceChecker<T> relevanceChecker;

    public MeanReciprocalRank() {
        this( new ExactMatchRelevanceChecker<T>(), DEFAULT_CUTOFF_RANK);
    }

    public MeanReciprocalRank(RelevanceChecker<T> relevanceChecker, int cutOffRank) {
        this.cutOffRank = cutOffRank;
        this.relevanceChecker = relevanceChecker;
    }

    /** {@inheritDoc} */
    public double eval(RankedResults<T>... queriesResults) {
        return eval(Arrays.asList(queriesResults), this.cutOffRank);
    }

    /** {@inheritDoc} */
    public double eval(List<RankedResults<T>> queriesResults) {
        return eval(queriesResults,  this.cutOffRank);
    }

    /** {@inheritDoc} */
    @Override
    public double eval(List<RankedResults<T>> queriesResults, int cutoffRank) {
        double sumReciprocalRanks = 0.0;
        int validQueryCount = 0;

        for (RankedResults<T> rankedResult : queriesResults) {
            Set<T> relevantDocs = rankedResult.getExpectedAnswers();
            if (relevantDocs == null || relevantDocs.isEmpty()) {
                // No ground truth for this query, skip it
                continue;
            }

            validQueryCount++;
            NavigableMap<Double, List<T>> descendingResults = rankedResult.getMatches().descendingMap();

            int rank = 1;
            boolean found = false;

            outerLoop:
            for (Map.Entry<Double, List<T>> entry : descendingResults.entrySet()) {
                List<T> docsAtScore = entry.getValue();

                for (T doc : docsAtScore) {
                    if (relevanceChecker.isRelevant(doc, relevantDocs)) {
                        // Found the first relevant document
                        sumReciprocalRanks += 1.0 / rank;
                        found = true;
                        break outerLoop;
                    }
                    rank++;
                }
            }

            if (!found) {
                // No relevant document found; reciprocal rank is 0
                sumReciprocalRanks += 0.0;
            }
        }

        // Avoid division by zero if validQueryCount is zero
        return validQueryCount > 0 ? sumReciprocalRanks / validQueryCount : 0.0;
    }
}
