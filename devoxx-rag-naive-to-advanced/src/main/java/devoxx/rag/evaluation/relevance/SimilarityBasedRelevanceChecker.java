package devoxx.rag.evaluation.relevance;

import dev.langchain4j.data.embedding.Embedding;
import devoxx.rag.similarity.CosineSimilarity;
import devoxx.rag.similarity.Similarity;

import java.util.Set;

/**
 * Check for Similarity.
 */
public class SimilarityBasedRelevanceChecker implements RelevanceChecker {

    static final double DEFAULT_THRESHOLD = .5d;

    /** function to compute similarity. */
    private final Similarity similarity;

    /** threshold for similarity. */
    private final double similarityThreshold;

    public SimilarityBasedRelevanceChecker() {
        this(new CosineSimilarity(), DEFAULT_THRESHOLD);
    }

    /** Constructor for SimilarityBasedRelevanceChecker.
     * @param similarity
     *     similarity
     * @param similarityThreshold
     *     similarity threshold
     */
    public SimilarityBasedRelevanceChecker(Similarity similarity, double similarityThreshold) {
        this.similarityThreshold = similarityThreshold;
        this.similarity = similarity;
    }

    @Override
    public boolean isRelevant(Embedding retrievedDoc, Set<Embedding> groundTruth) {
        for (Embedding truthDoc : groundTruth) {
            double result = similarity.compute(retrievedDoc, truthDoc);
            if (result >= similarityThreshold) {
                return true;
            }
        }
        return false;
    }
}
