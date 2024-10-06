package devoxx.rag.evaluation.relevance;

import java.util.Set;

/**
 * Default matcher for the relevance.
 */
public class ExactMatchRelevanceChecker<T> implements RelevanceChecker<T> {

    /** {@inheritDoc} */
    @Override
    public boolean isRelevant(T retrievedDoc, Set<T> groundTruth) {
        return groundTruth.contains(retrievedDoc);
    }

}
