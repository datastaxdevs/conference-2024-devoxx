package devoxx.rag.evaluation.relevance;

import dev.langchain4j.data.embedding.Embedding;

import java.util.Set;

/**
 * Evaluate how an embedding is relevant.
 */
public interface RelevanceChecker {

    /**
     * Check relevance of embedding.
     * @param retrievedDoc
     *      retrieve documentations
     * @param groundTruth
     *      retrieve documentations
     * @return
     *      evaluate if an embedding is relevant (for MMR algorithms)
     */
    boolean isRelevant(Embedding retrievedDoc, Set<Embedding> groundTruth);

}
