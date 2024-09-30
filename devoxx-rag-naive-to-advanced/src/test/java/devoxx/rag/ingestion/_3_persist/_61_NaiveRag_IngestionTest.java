package devoxx.rag.ingestion._3_persist;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Date;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;


public class _61_NaiveRag_IngestionTest extends AbstracDevoxxSampleTest {

    @Test
    public void shouldIngestDocument() throws Exception {

        // Parse Document
        Document document = loadDocumentText("johnny.txt");

        // QUICK QUICK
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);

        createCollectionRAG();
        getCollectionRAG().deleteAll();
        MessageDigest md = MessageDigest.getInstance("MD5");

        AtomicInteger counter = new AtomicInteger(0);
        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_GECKO))
                //.embeddingStore(new AstraDbEmbeddingStore(getCollectionRAG()))
                .embeddingStore(new InMemoryEmbeddingStore<>())
                .textSegmentTransformer(ts -> {
                    //ts.metadata().add("_id", "doc-johnny");
                    ts.metadata().add("document_id", "doc-johnny");
                    ts.metadata().add("document_format", "text");
                    ts.metadata().add("chunk_id", counter.incrementAndGet());
                    ts.metadata().add("index_date", new Date());
                    ts.metadata().add("granted_roles", List.of("USER", "ADMIN"));
                    ts.metadata().add("md5", hashString(ts.text()));
                    return ts;
                }).build().ingest(document);
    }

    public static String hashString(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hashedBytes = md.digest(input.getBytes());

            // Convert byte array into signum representation
            BigInteger no = new BigInteger(1, hashedBytes);

            // Convert message digest into hex value
            String hashtext = no.toString(16);
            while (hashtext.length() < 32) {
                hashtext = "0" + hashtext;
            }
            return hashtext;
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    private static ContentRetriever createContentRetriever(List<Document> documents) {

        // Here, we create and empty in-memory store for our documents and their embeddings.
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Here, we are ingesting our documents into the store.
        // Under the hood, a lot of "magic" is happening, but we can ignore it for now.
        EmbeddingStoreIngestor.ingest(documents, embeddingStore);

        // Lastly, let's create a content retriever from an embedding store.
        return EmbeddingStoreContentRetriever.from(embeddingStore);
    }
}