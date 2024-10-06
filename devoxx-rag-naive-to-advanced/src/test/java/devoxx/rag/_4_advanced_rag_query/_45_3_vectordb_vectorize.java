package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.Collection;
import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.DataAPIOptions;
import com.datastax.astra.client.model.CollectionIdTypes;
import com.datastax.astra.client.model.CollectionOptions;
import com.datastax.astra.client.model.Document;
import com.datastax.astra.client.model.SimilarityMetric;
import com.datastax.astra.internal.command.LoggingCommandObserver;
import com.datastax.astra.internal.utils.AnsiUtils;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import com.datastax.astra.langchain4j.store.embedding.EmbeddingSearchRequestAstra;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URISyntaxException;

public class _45_3_vectordb_vectorize extends AbstractDevoxxTest {

    private static final String COLLECTION_NAME = "quote";

    @Test
    public void should_vectorize() throws URISyntaxException {
        // Setup the store
        CollectionOptions collectionOptions = CollectionOptions
                .builder()
                .vectorSimilarity(SimilarityMetric.COSINE)
                .vectorize("nvidia","NV-Embed-QA")
                .build();
        Collection<Document> collection = new DataAPIClient(ASTRA_TOKEN,
                DataAPIOptions.builder().withObserver(new LoggingCommandObserver(AbstractDevoxxTest.class)).build())
                .getDatabase("https://3a2670a5-adbb-449e-b744-16d5182f5b70-us-east-2.apps.astra.datastax.com")
                .createCollection("vectorize_test", collectionOptions);
        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(collection);

        // Ingest documents
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);

        // Add and compute vectors on the SPOT
        embeddingStore.getCollection().deleteAll();
        embeddingStore.addAllVectorize(splitter.split(loadDocumentText("text/johnny.txt")));
        embeddingStore.addAllVectorize(splitter.split(loadDocumentText("text/shadow.txt")));
        embeddingStore.addAllVectorize(splitter.split(loadDocumentText("text/berlin.txt")));

        // Search
        EmbeddingSearchRequestAstra searchQuery = EmbeddingSearchRequestAstra.builderAstra()
                .maxResults(10)
                .minScore(0.1d)
                .queryVectorize("What is the Name of the HORSE ?")
                .build();
        embeddingStore.search(searchQuery).matches()
                .stream()
                .map(match -> match.embedded().text())
                .forEach(System.out::println);
    }



}
