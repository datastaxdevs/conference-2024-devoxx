package devoxx.rag.rrf;

import dev.langchain4j.data.segment.TextSegment;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implements Reciprocal Rank Fusion (RRF) to combine ranked results.
 */
public class RRFReranking  {

    private static final int DEFAULT_K = 60;

    /** RRF hyperparameter (typically 60). */
    private final int k;

    /**
     * Default Constructor.
     */
    public RRFReranking() {
        this(DEFAULT_K);
    }

    /**
     * Applying the RRF reranking algorithm.
     *
     * @param k
     *    The RRF hyperparameter.
     */
    public RRFReranking(int k) {
        this.k = k;
    }

    /**
     * Implements Reciprocal Rank Fusion (RRF) to combine ranked results.
     *
     * @param chunks
     *      List of ranked lists of chunks.
     * @return
     *      A map with document IDs and their combined RRF scores.
     */
    public Map<TextSegment, Double> reciprocalRankFusion(List<List<TextSegment>> chunks) {
        Map<TextSegment, Double> rrfScores = new HashMap<>();
        // Iterate over each ranked list
        for (List<TextSegment> ranking : chunks) {
            // Iterate through each document and calculate its RRF score based on its position
            for (int i = 0; i < ranking.size(); i++) {
                TextSegment docId = ranking.get(i);
                rrfScores.put(docId, rrfScores.getOrDefault(docId, 0.0) + 1.0 / (k + i + 1));
            }
        }
        return rrfScores;
    }
}
