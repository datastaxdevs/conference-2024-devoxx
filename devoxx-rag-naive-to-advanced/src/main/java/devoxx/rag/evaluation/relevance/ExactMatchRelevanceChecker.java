package devoxx.rag.evaluation.relevance;

import dev.langchain4j.data.embedding.Embedding;

import java.util.Set;

/**
 * Default matcher for the relevance.
 */
public class ExactMatchRelevanceChecker implements RelevanceChecker {

    /** {@inheritDoc} */
    @Override
    public boolean isRelevant(Embedding retrievedDoc, Set<Embedding> groundTruth) {
        return groundTruth.contains(retrievedDoc);
    }

}
