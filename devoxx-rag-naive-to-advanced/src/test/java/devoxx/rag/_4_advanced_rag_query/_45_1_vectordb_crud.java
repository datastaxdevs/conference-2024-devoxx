package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.DataAPIOptions;
import com.datastax.astra.internal.command.LoggingCommandObserver;
import com.datastax.astra.internal.utils.AnsiUtils;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class _45_1_vectordb_crud extends AbstractDevoxxTest {

    private static final String COLLECTION_NAME = "quote_prepopulated";

    @Test
    public void should_search_with_metadata() {
        String question = "We struggle all our life for nothing";
        Embedding questionEmbedding = getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed(question).content();
        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(new DataAPIClient(ASTRA_TOKEN,
                DataAPIOptions.builder().withObserver(new LoggingCommandObserver(AbstractDevoxxTest.class)).build())
                .getDatabase(ASTRA_API_ENDPOINT).getCollection(COLLECTION_NAME));

        EmbeddingSearchRequest searchQuery = EmbeddingSearchRequest.builder()
                .filter(new IsEqualTo("authors", "aristotle"))
                .maxResults(10)
                .minScore(0.1d)
                .queryEmbedding(questionEmbedding)
                .build();
        EmbeddingSearchResult<TextSegment> aristotleResults = embeddingStore.search(searchQuery);

        System.out.println(AnsiUtils.yellow("=========== ARISTOTLE ============"));
        aristotleResults.matches().forEach(match -> {
            System.out.println(AnsiUtils.cyan(BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP).toString()) + " - " + match.embedded().text());
        });

    }


}
