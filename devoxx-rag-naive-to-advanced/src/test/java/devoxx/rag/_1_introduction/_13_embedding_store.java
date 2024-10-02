package devoxx.rag._1_introduction;

import com.datastax.astra.client.Collection;
import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.model.Document;
import com.datastax.astra.client.model.SimilarityMetric;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _13_embedding_store extends AbstracDevoxxSampleTest {

    public static final String COLLECTION_NAME = "intro";

    @Test
    public void should_connect_store() {
        System.out.println(yellow("Connect Vector Database"));

        // Create Collection
        Collection<Document> vectorStore = new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .createCollection(COLLECTION_NAME, 5, SimilarityMetric.COSINE);
        System.out.println(cyan("[OK] ") + " Collection Created");

        // Insert a document
        vectorStore.deleteAll();
        Document document = new Document()
                .append("content", "Hello World")
                .vector(new float[] {.2f, .2f, .2f, .2f, .2f});
        vectorStore.insertOne(document);
        System.out.println(cyan("[OK] ") + " Document inserted");

        // With LangChain4J
        EmbeddingStore<TextSegment> embeddingStore = new AstraDbEmbeddingStore(vectorStore);
        embeddingStore.add(Embedding.from(new float[] {.2f, .2f, .2f, .2f, .2f}), TextSegment.from("Hello World"));
        System.out.println(cyan("[OK] ") + " Document inserted with store");
    }

    @Test
    public void deleteCollection() {
        System.out.println(yellow("Delete Collection"));
        new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .dropCollection(COLLECTION_NAME);
        System.out.println(cyan("[OK] ") + " Collection Deleted");
    }
}
