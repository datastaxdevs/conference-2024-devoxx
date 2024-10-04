package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.bm25.Bm25ScoringModel;
import devoxx.rag.bm25.Language;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

public class _46_reranking_bm25 extends AbstractDevoxxTest {

    private static final String COLLECTION_NAME = "quote";

    @Test
    public void should_search_in_vector_db() throws IOException {
        // I have to create a EmbeddingModel
        EmbeddingModel embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);
        // Embed the question
        String question = "We struggle all our life for nothing";
        Embedding questionEmbedding = embeddingModel.embed(question).content();
        // We need the store
        EmbeddingStore<TextSegment> embeddingStore = new AstraDbEmbeddingStore(getCollection(COLLECTION_NAME));
        // Build the Search Query
        EmbeddingSearchRequest searchQuery = EmbeddingSearchRequest.builder()
                .queryEmbedding(questionEmbedding)
                .maxResults(25)   // increase the number of users
                .minScore(0.1d)  // similarity score low to get more results
                .build();

        // Execute the request
        List<EmbeddingMatch<TextSegment>> matches = embeddingStore.search(searchQuery).matches();
        matches.stream().forEach(match -> {
            System.out.println("Similarity: " + BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP) + " - " + match.embedded().text());
        });

        // ReRanking
        List<TextSegment> chunks = matches.stream().map(EmbeddingMatch::embedded).toList();
        List<Double> scores = new Bm25ScoringModel(Language.ENGLISH).scoreAll(chunks, question).content();
        for (int i = 0; i < chunks.size(); i++) {
            System.out.println("BM25 Score: " + scores.get(i) + " - " + chunks.get(i).text());
        }
    }
}
