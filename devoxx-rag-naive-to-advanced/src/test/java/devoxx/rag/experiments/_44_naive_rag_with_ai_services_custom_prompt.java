package devoxx.rag.experiments;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;

public class _44_naive_rag_with_ai_services_custom_prompt {
    public static void main(String[] args) throws IOException, URISyntaxException {
        // ===============
        // INGESTION PHASE

        Document document = FileSystemDocumentLoader.loadDocument(
            Path.of("src/main/resources/cymbal-starlight-2024.pdf"),
            new ApachePdfBoxDocumentParser()
        );

        VertexAiEmbeddingModel embeddingModel = VertexAiEmbeddingModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .publisher("google")
            .modelName("text-embedding-004")
            .maxRetries(3)
            .build();

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor storeIngestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(DocumentSplitters.recursive(500, 100))
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .build();
        System.out.println("Chunking and embedding PDF...");
        storeIngestor.ingest(document);

        // ===============
        // RETRIEVAL PHASE

        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("gemini-1.5-flash")
            .maxOutputTokens(1000)
            .build();

        EmbeddingStoreContentRetriever retriever = //TODO change max results
            new EmbeddingStoreContentRetriever(embeddingStore, embeddingModel, 1);

        interface CarExpert {
            Result<String> ask(String question);
        }

        CarExpert expert = AiServices.builder(CarExpert.class)
            .chatLanguageModel(model)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .retrievalAugmentor(DefaultRetrievalAugmentor.builder()
                .contentInjector(DefaultContentInjector.builder()
                    .promptTemplate(PromptTemplate.from("""
                        You are an expert in car automotive, and you answer concisely.

                        Here is the question: {{userMessage}}

                        If you don't know the answer, reply that you don't know the answer.

                        Answer exclusively using the following information:
                        {{contents}}
                        """))
                    .build())
                .contentRetriever(retriever)
                .build())
            .build();

        System.out.println("Ready!\n");
        List.of(
            "What is the cargo capacity of Cymbal Starlight?",
            "What's the emergency roadside assistance phone number?",
            "Are there some repair kits available on that car?"
        ).forEach(query -> {
            Result<String> response = expert.ask(query);
            System.out.printf("%n=== %s === %n%n %s %n%n", query, response.content());
            System.out.println("SOURCE: " + response.sources().getFirst().textSegment().text());
        });
    }
}
