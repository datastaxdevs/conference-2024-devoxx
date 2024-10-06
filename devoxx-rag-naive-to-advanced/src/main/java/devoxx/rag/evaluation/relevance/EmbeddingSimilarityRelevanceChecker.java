package devoxx.rag.evaluation.relevance;

import dev.langchain4j.data.embedding.Embedding;
import devoxx.rag.similarity.CosineSimilarity;
import devoxx.rag.similarity.Similarity;

import java.util.Set;

/**
 * Check for Similarity, we are working with embeddings directly
 */
public class EmbeddingSimilarityRelevanceChecker implements RelevanceChecker<Embedding> {

    static final double DEFAULT_THRESHOLD = .5d;

    /** function to compute similarity. */
    private final Similarity similarity;

    /** threshold for similarity. */
    private final double similarityThreshold;

    /**
     * Constructor for SimilarityBasedRelevanceChecker.
     */
    public EmbeddingSimilarityRelevanceChecker() {
        this(new CosineSimilarity(), DEFAULT_THRESHOLD);
    }

    /** Constructor for SimilarityBasedRelevanceChecker.
     * @param similarity
     *     similarity
     * @param similarityThreshold
     *     similarity threshold
     */
    public EmbeddingSimilarityRelevanceChecker(Similarity similarity, double similarityThreshold) {
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
