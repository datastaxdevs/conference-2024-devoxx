package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;
import static devoxx.rag._3_advanced_rag_ingestion._37_hypothetical_questions_embedding.getEmbeddingStore;

public class _43_query_compression extends AbstractDevoxxTest {

    @Test
    public void shouldTestQueryCompression() {

        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
            .embeddingStore(getEmbeddingStore())
            .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_TEXT))
            .build();

        VertexAiGeminiChatModel chatModel = VertexAiGeminiChatModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("gemini-2.0-flash-001")
            .listeners(List.of(new ChatModelListener() {
                @Override
                public void onResponse(ChatModelResponseContext responseContext) {
                    System.out.println("REFORMULATED: " + yellow(responseContext.response().aiMessage().text()) + "\n");
                }
            }))
            .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(retriever)
                .queryTransformer(new CompressingQueryTransformer(chatModel))
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_FLASH))
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        List.of(
            "What is the capital of Germany?",
            "Tell me more about its geographical situation",
            "How many people live there?"
        ).forEach(query -> {
            System.out.println("\n=== " + cyan(query) + " ===\n");

            String answer = assistant.answer(query);

            System.out.println(answer);
        });
    }
}