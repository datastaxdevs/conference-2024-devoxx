package devoxx.rag._vectordb;

import com.datastax.astra.client.model.Document;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.Quote;
import devoxx.rag.TestSupport;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;
import static devoxx.rag.Utilities.loadQuotes;

@Slf4j
class _55_AstraVectorTest extends TestSupport {

    @Test
    void shouldIngestDocuments() throws IOException {
        getCollectionQuote().deleteAll();
        EmbeddingModel embeddingModel = getEmbeddingModelGecko();

        loadQuotes("philo_quotes.json")       // extraction
                .stream()
                .map(quote -> mapAsDocument(embeddingModel, quote))// no chunking (single sentences)
                .forEach(doc -> {getCollectionQuote().insertOne(doc);
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                });

        //createCollectionQuote().insertMany(
        //  loadQuotes("philo_quotes.json")       // extraction
        //     .stream()
        //     .map(quote -> mapAsDocument(embeddingModel, quote))// no chunking (single sentences)
        //     .toList()
        //);
    }

    @Test
    void langchain4jEmbeddingStore() {
        // I have to create a EmbeddingModel
        EmbeddingModel embeddingModel = getEmbeddingModelGecko();

        // Embed the question
        Embedding questionEmbedding = embeddingModel.embed("We struggle all our life for nothing").content();

        // We need the store
        EmbeddingStore<TextSegment> embeddingStore = new AstraDbEmbeddingStore(getCollectionQuote());

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
        log.info("Mapping quote: {}", quote.rowId());
        return new Document(quote.rowId())
                .append("content", quote.body())
                .append("authors", quote.author())
                .append("tags", quote.tags())
                .vector(embeddingModel.embed(quote.body()).content().vector());
    }
}
