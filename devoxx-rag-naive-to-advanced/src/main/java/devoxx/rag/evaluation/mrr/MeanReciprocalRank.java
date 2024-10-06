package devoxx.rag.evaluation.mrr;

import dev.langchain4j.data.embedding.Embedding;
import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.RankedResultsEvaluator;
import devoxx.rag.evaluation.relevance.ExactMatchRelevanceChecker;
import devoxx.rag.evaluation.relevance.RelevanceChecker;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;

/**
 * Proposition for an implementation of MMR
 */
public class MeanReciprocalRank implements RankedResultsEvaluator {

    private static final int DEFAULT_CUTOFF_RANK = 15;

    private final int cutOffRank;

    /** Default relevant checker. */
    private static final RelevanceChecker DEFAULT_RELEVANCE_CHECKER =
            new ExactMatchRelevanceChecker();

    private final RelevanceChecker relevanceChecker;

    public MeanReciprocalRank() {
        this(DEFAULT_RELEVANCE_CHECKER, DEFAULT_CUTOFF_RANK);
    }

    public MeanReciprocalRank(RelevanceChecker relevanceChecker, int cutOffRank) {
        this.cutOffRank = cutOffRank;
        this.relevanceChecker = relevanceChecker;
    }

    /** {@inheritDoc} */
    public double eval(RankedResults... queriesResults) {
        return eval(Arrays.asList(queriesResults), this.cutOffRank);
    }

    /** {@inheritDoc} */
    public double eval(List<RankedResults> queriesResults) {
        return eval(queriesResults,  this.cutOffRank);
    }

    /** {@inheritDoc} */
    @Override
    public double eval(List<RankedResults> queriesResults, int cutoffRank) {
        System.out.println(queriesResults.get(0).getResults().size());

        double sumReciprocalRanks = 0.0;
        int validQueryCount = 0;

        for (RankedResults rankedResult : queriesResults) {
            Set<Embedding> relevantDocs = rankedResult.getGroundTruth();

            if (relevantDocs == null || relevantDocs.isEmpty()) {
                // No ground truth for this query, skip it
                continue;
            }

            validQueryCount++;
            NavigableMap<Double, List<Embedding>> descendingResults = rankedResult.getResults().descendingMap();

            int rank = 1;
            boolean found = false;

            outerLoop:
            for (Map.Entry<Double, List<Embedding>> entry : descendingResults.entrySet()) {
                List<Embedding> docsAtScore = entry.getValue();

                for (Embedding doc : docsAtScore) {
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
