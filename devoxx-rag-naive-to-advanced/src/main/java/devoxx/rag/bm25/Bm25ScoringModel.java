package devoxx.rag.bm25;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;

import java.util.List;

/**
 * Implementation of a ReRanking Scoring Model based on BM25.
 */
public class Bm25ScoringModel implements ScoringModel {

    /** Default value for the k1 parameter. */
    private static final double TERM_FREQUENCY_SCALING_FACTOR = 1.5;

    /** Default value for the b parameter. */
    private static final double DOCUMENT_LENGTH_SCALING_FACTOR = 0.75;

    private final Language language;

    private final double termFrequencyScalingFactor;

    private final double documentLengthScalingFactor;

    /** Constructor */
    public Bm25ScoringModel(Language language) {
        this(language, TERM_FREQUENCY_SCALING_FACTOR, DOCUMENT_LENGTH_SCALING_FACTOR);
    }

    /** Constructor */
    public Bm25ScoringModel(Language language, double termFrequencyScalingFactor, double documentLengthScalingFactor) {
        this.language = language;
        this.termFrequencyScalingFactor = termFrequencyScalingFactor;
        this.documentLengthScalingFactor = documentLengthScalingFactor;
    }

    /** {@inheritDoc} */
    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        BM25 bm25 = new BM25(segments.stream().map(TextSegment::text).toList(),
                language,
                TERM_FREQUENCY_SCALING_FACTOR,
                DOCUMENT_LENGTH_SCALING_FACTOR);
        List<java.util.Map.Entry<Integer, Double>> entries = bm25.search(query);
        return Response.from(entries.stream().map(java.util.Map.Entry::getValue).toList());
    }

    /**
     * Gets language
     *
     * @return value of language
     */
    public Language getLanguage() {
        return language;
    }

    /**
     * Gets termFrequencyScalingFactor
     *
     * @return value of termFrequencyScalingFactor
     */
    public double getTermFrequencyScalingFactor() {
        return termFrequencyScalingFactor;
    }

    /**
     * Gets documentLengthScalingFactor
     *
     * @return value of documentLengthScalingFactor
     */
    public double getDocumentLengthScalingFactor() {
        return documentLengthScalingFactor;
    }
}
