package devoxx.rag._vectordb;

import com.datastax.astra.client.Database;
import com.datastax.astra.client.model.CollectionOptions;
import com.datastax.astra.client.model.SimilarityMetric;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import devoxx.rag.TestSupport;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.File;
import java.nio.file.Path;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * ASTRA can now compute the embeddings for you. This is a simple example of how to use ASTRA to compute embeddings.
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@EnabledIfEnvironmentVariable(named = "ASTRA_DB_APPLICATION_TOKEN", matches = "Astra.*")
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = "sk.*")
public class GettingStartedGuideTestIT extends TestSupport {

    static Database astraDatabase;
    static AstraDbEmbeddingStore storeJohnny;
    static AstraDbEmbeddingStore storeShadow;
    static EmbeddingModel openAiEmbeddingModel;

    @BeforeAll
    public static void initStoreForTests() {
        astraDatabase = AstraDBTestSupport.createDbIfNotExist(AstraDBTestSupport.TEST_DB);
        /*
         * Alternative if you know the database endpoint
         *  astraDatabase = new DataAPIClient(ASTRA_TOKEN, DataAPIOptions
         *      .builder()
         *      .logRequests().build())
         *      .getDatabase("https://astra.datastax.com/api/rest/v1/keyspaces");
         */
        storeJohnny = new AstraDbEmbeddingStore(
           astraDatabase.createCollection("store_johnny", CollectionOptions
                .builder()
                .vector(1536, SimilarityMetric.COSINE)
                .vectorize("openai", "text-embedding-3-small")
                .build()));
        storeJohnny.clear();

        storeShadow = new AstraDbEmbeddingStore(
                astraDatabase.createCollection("store_shadow", CollectionOptions
                        .builder()
                        .vector(1536, SimilarityMetric.COSINE)
                        .vectorize("openai", "text-embedding-3-small")
                        .build()));
        storeShadow.clear();

        // Access the embedding Model



    }

    @Test
    @Order(1)
    public void should_search_naive_rag() {
        openAiEmbeddingModel = getEmbeddingModelGecko();
        ingestDocument("johnny.txt", storeJohnny);
        ingestDocument("shadow.txt", storeShadow);

        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "Base your answer on the following information:\n"
                        + "{{rag-context}}");

        String question = "Who is Johnny?";
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = storeJohnny
                .search(EmbeddingSearchRequest.builder()
                        //.filter(metadataKey("document_format").isEqualTo("text"))
                        .queryEmbedding(openAiEmbeddingModel.embed(question).content())
                        .minScore(0.5)
                        .maxResults(2)
                        .build()).matches();

        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("rag-context", relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(Collectors.joining("\n\n")));
        log.info("{}", variables);
        Prompt prompt = promptTemplate.apply(variables);

        // See an answer from the model
        log.info( getChatLanguageModelChatBison()
                .generate(prompt.toUserMessage()).content().text());
    }

    @Test
    @Order(2)
    public void should_search_content_retriever() {

            ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(storeJohnny)
                    .embeddingModel(getEmbeddingModelGecko())
                    .maxResults(2)
                    .minScore(0.5)
                    .build();

            // configuring it to use the components we've created above.
            Assistant ai = AiServices.builder(Assistant.class)
                    .contentRetriever(contentRetriever)
                    .chatLanguageModel(getChatLanguageModelChatBison())
                    //.chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                    .build();

            String response = ai.answer("Who is Johnny?");
            System.out.println(response);
    }

    @Test
    @Order(3)
    public void should_search_advanced_rag() {
        // Our guy for advanced RAG
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new MyRouter())
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModelChatBison())
                .build();

        System.out.println(assistant.answer("Give me the name of the horse"));
        System.out.println(assistant.answer("Give me the name of the dog"));
    }

    private static class MyRouter implements QueryRouter {

        @Override
        public Collection<ContentRetriever> route(Query query) {

            if (query.text().contains("horse")) {
                return List.of(EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeJohnny)
                        .embeddingModel(openAiEmbeddingModel)
                        .maxResults(2)
                        .minScore(0.6)
                        .build());
            }

            if (query.text().contains("dog")) {
                return  List.of(EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeShadow)
                        .embeddingModel(openAiEmbeddingModel)
                        .maxResults(2)
                        .minScore(0.6)
                        .build());
            }

            return List.of();
        }

    }


    private static void ingestDocument(String docName, EmbeddingStore<TextSegment> store) {
        Path path = new File(Objects.requireNonNull(GettingStartedGuideTestIT.class
                .getResource("/" + docName)).getFile()).toPath();
        Document document = FileSystemDocumentLoader
                .loadDocument(path, new TextDocumentParser());
        DocumentSplitter splitter = DocumentSplitters
                .recursive(300, 0);
        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(openAiEmbeddingModel)
                .embeddingStore(store).build().ingest(document);
    }
}
