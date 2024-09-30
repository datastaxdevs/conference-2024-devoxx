package devoxx.rag;

import com.datastax.astra.client.Collection;
import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.model.Document;
import com.datastax.astra.internal.command.LoggingCommandObserver;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiStreamingChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static com.datastax.astra.client.model.SimilarityMetric.COSINE;
import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static devoxx.rag.Utilities.ASTRA_API_ENDPOINT;
import static devoxx.rag.Utilities.ASTRA_TOKEN;
import static devoxx.rag.Utilities.EMBEDDING_DIMENSION;
import static devoxx.rag.Utilities.GCP_PROJECT_ENDPOINT;
import static devoxx.rag.Utilities.GCP_PROJECT_PUBLISHER;

/**
 * Abstract Class for different tests and use cases to share configuration
 */
public abstract class AbstracDevoxxSampleTest {


    // ------------------------------------------------------------
    //                           GEMINI STUFF
    // ------------------------------------------------------------

    /** GEMINI SETTINGS. */
    protected final String GCP_PROJECT_ID        = "devoxxfrance";
    protected final String GCP_PROJECT_LOCATION  = "us-central1";
    protected final String MODEL_GEMINI_PRO       = "gemini-pro";
    protected final String MODEL_GEMINI_FLASH     = "gemini-flash";
    protected final String MODEL_GEMINI_EMBEDDING = "gemini-light";
    protected final String MODEL_EMBEDDING_GECKO  = "textembedding-gecko@001";

    /** Create a the chat model. */
    protected ChatLanguageModel getChatLanguageModel(final String modelName) {
        return VertexAiGeminiChatModel.builder()
                .project(GCP_PROJECT_ID)
                .location(GCP_PROJECT_LOCATION)
                .modelName(modelName)
                .build();
    }

    /** Create a Streaming Chat Model. */
    protected StreamingChatLanguageModel getChatLanguageModelStreaming(final String modelName) {
        return VertexAiGeminiStreamingChatModel.builder()
                .project(GCP_PROJECT_ID)
                .location(GCP_PROJECT_LOCATION)
                .modelName(modelName)
                .build();
    }

    /** Create an Embedding model. */
    protected EmbeddingModel getEmbeddingModel(final String modelName) {
        return VertexAiEmbeddingModel.builder()
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)
                .publisher(GCP_PROJECT_PUBLISHER)
                .modelName(modelName)
                .build();
    }


    // ------------------------------------------------------------
    //                ASTRA / CASSANDRA STORE STUFF
    // ------------------------------------------------------------

    public Collection<Document> getCollectionQuote() {
        Collection<Document> col =  new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .getCollection("quote_store", Document.class);
        col.registerListener("logger", new LoggingCommandObserver(AbstracDevoxxSampleTest.class));
        return col;
    }

    public Collection<Document> createCollectionQuote() {
        return new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .createCollection("quote_store", EMBEDDING_DIMENSION, COSINE);
    }

    public Collection<Document> createCollectionRAG() {
        return new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .createCollection("rag_store", EMBEDDING_DIMENSION, COSINE);
    }

    public Collection<Document> getCollectionRAG() {
        return new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .getCollection("rag_store");
    }

    // ------------------------------------------------------------
    //               RAG STUFF
    // ------------------------------------------------------------


    private static void ingestDocument(String docName, EmbeddingModel model, EmbeddingStore<TextSegment> store) {
        Path path = new File(Objects.requireNonNull(AbstracDevoxxSampleTest.class
                .getResource("/" + docName)).getFile()).toPath();
        dev.langchain4j.data.document.Document document = FileSystemDocumentLoader
                .loadDocument(path, new TextDocumentParser());
        DocumentSplitter splitter = DocumentSplitters
                .recursive(300, 20);

        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(model)
                .embeddingStore(store).build().ingest(document);
    }

    protected ContentRetriever createRetriever(String fileName) {
        URL fileURL = getClass().getResource(fileName);
        Path path = new File(fileURL.getFile()).toPath();
        dev.langchain4j.data.document.Document document = FileSystemDocumentLoader
                .loadDocument(path, new TextDocumentParser());
        DocumentSplitter splitter = DocumentSplitters
                .recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(getEmbeddingModel(MODEL_EMBEDDING_GECKO).embedAll(segments).content(), segments);

        //ingestDocument(fileName, getEmbeddingModel(MODEL_EMBEDDING_GECKO), embeddingStore);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_GECKO))
                .maxResults(2)
                .minScore(0.6)
                .build();
    }

    protected EmbeddingStoreContentRetriever.EmbeddingStoreContentRetrieverBuilder createRetrieverBuilder(String fileName) {
        List<TextSegment> segments = DocumentSplitters
                .recursive(300, 20)
                .split(loadDocument(new File(Objects.requireNonNull(getClass()
                                .getResource(fileName))
                        .getFile()).toPath(), new TextDocumentParser()));
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(getEmbeddingModel(MODEL_EMBEDDING_GECKO).embedAll(segments).content(), segments);
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_GECKO))
                .maxResults(2)
                .minScore(0.6);
    }

    // ------------------------------------------------------------
    //            DISPLAY STUFF
    // ------------------------------------------------------------

    /**
     * Utilities function to show the results in the console
     *
     * @param response
     *      AI Response
     */
    protected static void prettyPrint(Response<AiMessage> response) {
        System.out.println(cyan("===================================================="));
        System.out.println(cyan("===                  RESPONSE                      ="));
        System.out.println(cyan("====================================================\n"));
        System.out.println(yellow(response.content().text().replaceAll("\\n\\n", "\\n")));
        System.out.println();
        if (response.finishReason()!=null) {
            System.out.println("Finish Reason : " + cyan(response.finishReason().toString()));
        }
        if (response.tokenUsage()!=null) {
            System.out.println("Tokens Input  : " + cyan(String.valueOf(response.tokenUsage().inputTokenCount())));
            System.out.println("Tokens Output : " + cyan(String.valueOf(response.tokenUsage().outputTokenCount())));
            System.out.println("Tokens Total  :  " + cyan(String.valueOf(response.tokenUsage().totalTokenCount())));
        }
    }

    /** Streaming handler to log results. */
    public static class PrettyPrintStreamingResponseHandler
            implements StreamingResponseHandler<AiMessage> {
       @Override
       public void onNext(String s) { System.out.println(s); }
       @Override
       public void onComplete(Response<AiMessage> response) {  prettyPrint(response);}
       @Override
       public void onError(Throwable throwable) { System.out.println("Error : " + throwable.getMessage());}
    }

    protected static String formatLongString(String input) {
        int limit = 100;
        StringBuilder result = new StringBuilder();
            int start = 0;

            while (start < input.length()) {
                int end = Math.min(start + limit, input.length());

                // Move the end back to the last space within the limit to avoid breaking words
                if (end < input.length() && input.charAt(end) != ' ') {
                    int lastSpace = input.lastIndexOf(' ', end);
                    if (lastSpace > start) {
                        end = lastSpace;
                    }
                }

                result.append(input, start, end).append("\n");
                start = end + 1;  // Skip the space after the newline
            }
        return result.toString();
    }

    public dev.langchain4j.data.document.Document loadDocumentText(String fileName) {
        Path path = new File(Objects.requireNonNull(getClass().getResource("/" + fileName)).getFile()).toPath();
        return FileSystemDocumentLoader.loadDocument(path, new TextDocumentParser());
    }










}
