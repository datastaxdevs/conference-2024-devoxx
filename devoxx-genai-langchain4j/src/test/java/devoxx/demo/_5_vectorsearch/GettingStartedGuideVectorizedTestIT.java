package devoxx.demo._5_vectorsearch;

import com.datastax.astra.client.Database;
import com.datastax.astra.client.model.CollectionOptions;
import com.datastax.astra.client.model.SimilarityMetric;
import com.datastax.astra.langchain4j.rag.AstraVectorizeContentRetriever;
import com.datastax.astra.langchain4j.rag.AstraVectorizeIngestor;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import com.datastax.astra.langchain4j.store.embedding.EmbeddingSearchRequestAstra;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import devoxx.demo.utils.AbstractDevoxxTestSupport;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * ASTRA can now compute the embeddings for you. This is a simple example of how to use ASTRA to compute embeddings.
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@EnabledIfEnvironmentVariable(named = "ASTRA_DB_APPLICATION_TOKEN", matches = "Astra.*")
public class GettingStartedGuideVectorizedTestIT extends AbstractDevoxxTestSupport {

    /**
     * Different Embedding Stores topologies
     */
    static Database astraDatabase;
    static AstraDbEmbeddingStore embeddingStoreVectorizeNVidia;

    @BeforeAll
    public static void initStoreForTests() {
        astraDatabase = devoxx.demo._5_vectorsearch.AstraDBTestSupport.createDbIfNotExist(devoxx.demo._5_vectorsearch.AstraDBTestSupport.TEST_DB_vectorize);

        /*
         * An embedding store that compute the embedding for you on the fly without
         * the need of a embedding model. It is done at database level for you.
         */
        embeddingStoreVectorizeNVidia = new AstraDbEmbeddingStore(
                astraDatabase.createCollection("store_with_nvidia", CollectionOptions
                        .builder()
                        .vector(1024, SimilarityMetric.COSINE)
                        .vectorize("nvidia", "NV-Embed-QA")
                        .build()));

        // Empty Store to Start
        //embeddingStoreVectorizeNVidia.clear();
    }

    private static void ingestDocument(String docName, AstraDbEmbeddingStore store) {

        // Load the document
        Document inputDocument = FileSystemDocumentLoader.loadDocument(
                new File(devoxx.demo._5_vectorsearch.AstraDBTestSupport.class.getResource("/" + docName)
                        .getFile())
                        .toPath(),
                new TextDocumentParser());

        // Create some metadata
        SimpleDateFormat SDF = new SimpleDateFormat("yyyy-MM-dd");
        String ingestionDate = SDF.format(new Date());

        // Ingestion
        AstraVectorizeIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .documentTransformer(doc -> {
                    doc.metadata().put("insertion_date", ingestionDate);
                    doc.metadata().put("doc_name", docName);
                    return doc;
                })
                .embeddingStore(store)
                .build()
                .ingest(inputDocument);
    }

    @Test
    @Order(1)
    public void should_ingest_documents() {
        ingestDocument("johnny.txt", embeddingStoreVectorizeNVidia);
    }

    @Test
    @Order(2)
    public void should_search_direct() {
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStoreVectorizeNVidia
                .search(EmbeddingSearchRequestAstra.builderAstra()
                        .queryVectorize("Who is Johnny?")
                        .maxResults(10)
                        .minScore(0.5)
                        .build()).matches();
        assertThat(relevantEmbeddings).isNotEmpty();

        // Building a RAG CONTEXT
        relevantEmbeddings.forEach(match -> log.info("Match: {}", match.embedded()));
        String ragContext = relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(Collectors.joining("\n\n"));
        log.info("Rag Context {}", ragContext);
    }

    @Test
    @Order(3)
    public void should_search_with_content_retriever() {
        devoxx.demo._5_vectorsearch.Assistant ai = AiServices.builder(devoxx.demo._5_vectorsearch.Assistant.class)
                .contentRetriever(
                        AstraVectorizeContentRetriever.builder()
                        .embeddingStore(embeddingStoreVectorizeNVidia)
                        .filter(new IsEqualTo("file_name", "johnny.txt"))
                        .maxResults(5)
                        .minScore(0.2)
                        .build())
                .chatLanguageModel(getChatLanguageModelChatBison())
                .build();
        System.out.println(ai.answer("Give me the name of the HORSE"));
    }

    @Test
    @Order(4)
    public void should_search_advanced_rag() {

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                // Preprocessing
                //.queryRouter()
                //.queryTransformer()
                .contentRetriever(AstraVectorizeContentRetriever.builder()
                        .embeddingStore(embeddingStoreVectorizeNVidia)
                        .filter(new IsEqualTo("file_name", "johnny.txt"))
                        .maxResults(5)
                        .minScore(0.2)
                        .build())
                // Post Processir
                //.contentAggregator()
                .build();

        devoxx.demo._5_vectorsearch.Assistant ai = AiServices.builder(devoxx.demo._5_vectorsearch.Assistant.class)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModelChatBison())
                .build();
        System.out.println(ai.answer("Give me the name of the HORSE"));
    }
}
