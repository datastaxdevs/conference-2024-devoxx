package devoxx.rag._3_advanced_rag_ingestion;

import com.datastax.astra.client.model.Document;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Quote;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;

@Slf4j
class _39_custom_ingestion extends AbstractDevoxxTest {

    static final String COLLECTION_NAME = "quote";

    @Test
    void shouldIngestDocuments() throws IOException {
        createCollection(COLLECTION_NAME, MODEL_EMBEDDING_DIMENSION);
        EmbeddingModel embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);
        getCollection(COLLECTION_NAME).deleteAll();
        List<Document> docs = loadQuotes("/json/philo_quotes.json")       // extraction
                .stream()
                .map(quote -> mapAsDocument(embeddingModel, quote))// no chunking (single sentences);
                .toList();
        getCollection(COLLECTION_NAME).insertMany(docs);
    }

    @Test
    void langchain4jEmbeddingStore() {
        // I have to create a EmbeddingModel
        EmbeddingModel embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);

        // Embed the question
        Embedding questionEmbedding = embeddingModel.embed("We struggle all our life for nothing").content();

        // We need the store
        EmbeddingStore<TextSegment> embeddingStore = new AstraDbEmbeddingStore(getCollection(COLLECTION_NAME));

        // Query with a filter(2)
        log.info("Querying with filter");
        embeddingStore.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(questionEmbedding)
                        .filter(metadataKey("authors").isEqualTo("aristotle"))
                        .maxResults(3).minScore(0.1d).build())
                .matches()
                .stream().map(r -> r.embedded().text())
                .forEach(System.out::println);
    }


    Document mapAsDocument(EmbeddingModel embeddingModel , Quote quote) {
        return new Document(quote.rowId())
                .append("content", quote.body())
                .append("authors", quote.author())
                .append("tags", quote.tags())
                .vector(embeddingModel.embed(quote.body()).content().vector());
    }
}
