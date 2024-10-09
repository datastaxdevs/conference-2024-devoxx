package devoxx.rag._4_advanced_rag_query;

import com.google.cloud.vertexai.api.Schema;
import com.google.cloud.vertexai.api.Type;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _41_3_query_routing_with_llm extends AbstractDevoxxTest {

    @Test
    public void testQueryRouting() {
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new LlmRouter())
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
                .build();

        System.out.println(assistant.answer("Give me the name of the horse"));
        System.out.println(assistant.answer("Give me the name of the dog"));
    }

    /**
     * Custom Router
     */
    private static class LlmRouter extends AbstractDevoxxTest implements QueryRouter  {

        @Override
        public Collection<ContentRetriever> route(Query query) {
            VertexAiGeminiChatModel llmRouter = VertexAiGeminiChatModel.builder()
                .project(System.getenv("GCP_PROJECT_ID"))
                .location(System.getenv("GCP_LOCATION"))
                .modelName(MODEL_GEMINI_FLASH)
                .maxRetries(5)
                .responseSchema(Schema.newBuilder()
                    .setType(Type.STRING)
                    .addEnum("dog")
                    .addEnum("horse")
                    .build())
                .build();

            Response<AiMessage> classificationResponse = llmRouter.generate(
                SystemMessage.from("""
                    You must classify the user's message into two buckets:
                    Return "dog" if the message about dogs.
                    Return "horse" if the message about horses.
                    """),
                UserMessage.from(query.text())
            );

            String category = classificationResponse.content().text();

            System.out.println(yellow("-> Category recognized: " + category));

            if ("horse".equals(category)) {
                return List.of(createRetriever("/text/johnny.txt"));
            } else if ("dog".equals(category)) {
                return  List.of(createRetriever("/text/shadow.txt"));
            }

            return Collections.emptyList();
        }
    }
}