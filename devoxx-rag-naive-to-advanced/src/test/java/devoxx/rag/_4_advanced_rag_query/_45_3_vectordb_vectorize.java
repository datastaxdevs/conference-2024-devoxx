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
    public void should_crud_with_collection() {
        String question = "We struggle all our life for nothing";
        Embedding questionEmbedding = getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed(question).content();
        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(getCollection(COLLECTION_NAME));
    }

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

    @Test
    public void should_vectorize() throws URISyntaxException {
        // Setup the store
        CollectionOptions collectionOptions = CollectionOptions
                .builder()
                .vectorSimilarity(SimilarityMetric.COSINE)
                .defaultIdType(CollectionIdTypes.UUID)
                .vectorize("nvidia","NV-Embed-QA").build();
        Collection<Document> collection = new DataAPIClient(ASTRA_TOKEN)
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
                .stream().map(match -> match.embedded().text())
                .forEach(System.out::println);
    }



}
