package devoxx.demo.utils;

import com.datastax.astra.client.Collection;
import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.model.Document;
import com.datastax.astra.internal.command.LoggingCommandObserver;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.language.LanguageModel;
import dev.langchain4j.model.vertexai.VertexAiChatModel;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiLanguageModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.File;
import java.util.List;
import java.util.Objects;

import static com.datastax.astra.client.model.SimilarityMetric.COSINE;
import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static devoxx.demo.devoxx.Utilities.ASTRA_API_ENDPOINT;
import static devoxx.demo.devoxx.Utilities.ASTRA_TOKEN;
import static devoxx.demo.devoxx.Utilities.EMBEDDING_DIMENSION;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ENDPOINT;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ID;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_LOCATION;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_PUBLISHER;

public class AbstractDevoxxTestSupport {

    protected LanguageModel getLanguageModel(final String modelName) {
        return VertexAiLanguageModel.builder()
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)
                .publisher(GCP_PROJECT_PUBLISHER)
                .modelName(modelName)
                .build();
    }

    protected EmbeddingModel getEmbeddingModel(final String modelName) {
        return VertexAiEmbeddingModel.builder()
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)
                .publisher(GCP_PROJECT_PUBLISHER)
                .modelName(modelName)
                .build();
    }

    protected ChatLanguageModel getChatLanguageModel(final String modelName) {
        return VertexAiChatModel.builder()
                .publisher(GCP_PROJECT_PUBLISHER)
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)
                .modelName(modelName)
                .temperature(0.7)
                .topK(3)
                .topP(.8)
                .maxRetries(3)
                .maxOutputTokens(2000)
                .build();
    };

    protected ChatLanguageModel getChatLanguageModelChatBison() {
        return getChatLanguageModel("chat-bison");
    }

    protected LanguageModel getLanguageModelTextBison() {
        return getLanguageModel("text-bison");
    }

    protected EmbeddingModel getEmbeddingModelGecko() {
        return getEmbeddingModel("textembedding-gecko@001");
    }

    public Collection<Document> getCollectionQuote() {
        Collection<Document> col =  new DataAPIClient(ASTRA_TOKEN)
                .getDatabase(ASTRA_API_ENDPOINT)
                .getCollection("quote_store", Document.class);
        col.registerListener("logger", new LoggingCommandObserver(AbstractDevoxxTestSupport.class));
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

    protected ContentRetriever createRetriever(String fileName) {
        List<TextSegment> segments = DocumentSplitters
                .recursive(300, 0)
                .split(loadDocument(new File(Objects.requireNonNull(getClass()
                                .getResource(fileName))
                        .getFile()).toPath(), new TextDocumentParser()));
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(getEmbeddingModelGecko().embedAll(segments).content(), segments);
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(getEmbeddingModelGecko())
                .maxResults(2)
                .minScore(0.6)
                .build();
    }

    protected EmbeddingStoreContentRetriever.EmbeddingStoreContentRetrieverBuilder createRetrieverBuilder(String fileName) {
        List<TextSegment> segments = DocumentSplitters
                .recursive(300, 0)
                .split(loadDocument(new File(Objects.requireNonNull(getClass()
                                .getResource(fileName))
                        .getFile()).toPath(), new TextDocumentParser()));
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(getEmbeddingModelGecko().embedAll(segments).content(), segments);
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(getEmbeddingModelGecko())
                .maxResults(2)
                .minScore(0.6);
    }





}
