package devoxx.rag.evaluation.relevance;

import java.util.Set;

/**
 * Evaluate how an embedding is relevant.
 *
 * @param <T>
 *     item to validates, it could be the embeddings or the text chunks depending on the use case
 *     some user would like to override with their own structure,
 */
public interface RelevanceChecker<T> {

    /**
     * Check relevance of embedding.
     * @param retrievedDoc
     *      retrieve documentations
     * @param groundTruth
     *      retrieve documentations
     * @return
     *      evaluate if an embedding is relevant (for MMR algorithms)
     */
    boolean isRelevant(T retrievedDoc, Set<T> groundTruth);

}
