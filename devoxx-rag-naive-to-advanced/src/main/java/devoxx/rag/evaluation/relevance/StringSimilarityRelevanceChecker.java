package devoxx.rag.evaluation.relevance;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.internal.ValidationUtils;
import dev.langchain4j.model.embedding.EmbeddingModel;
import devoxx.rag.similarity.CosineSimilarity;
import devoxx.rag.similarity.Similarity;

import java.util.Set;

/**
 * Check for Similarity, we are working with embeddings directly
 */
public class StringSimilarityRelevanceChecker implements RelevanceChecker<String> {

    static final double DEFAULT_THRESHOLD = .5d;

    /** function to compute similarity. */
    private final Similarity similarity;

    /** threshold for similarity. */
    private final double similarityThreshold;

    private final EmbeddingModel embeddingModel;

    public StringSimilarityRelevanceChecker(EmbeddingModel embeddingModel) {
        this(embeddingModel, new CosineSimilarity(), DEFAULT_THRESHOLD);
    }

    /**
     * onstructor for SimilarityBasedRelevanceChecker.
     *
     * @param similarity
     *     similarity
     * @param similarityThreshold
     *     similarity threshold
     */
    public StringSimilarityRelevanceChecker(EmbeddingModel embeddingModel, Similarity similarity, double similarityThreshold) {
        ValidationUtils.ensureNotNull(embeddingModel, "embeddingModel");
        ValidationUtils.ensureNotNull(similarity, "similarity");
        this.embeddingModel      = embeddingModel;
        this.similarityThreshold = similarityThreshold;
        this.similarity          = similarity;
    }

    @Override
    public boolean isRelevant(String retrievedDoc, Set<String> groundTruth) {
        Embedding embeddingRetrievedDoc = embeddingModel.embed(retrievedDoc).content();
        for (String truthDoc : groundTruth) {
            Embedding embeddingTruthDoc= embeddingModel.embed(truthDoc).content();
            double result = similarity.compute(embeddingRetrievedDoc, embeddingTruthDoc);
            if (result >= similarityThreshold) {
                return true;
            }
        }
        return false;
    }
}
