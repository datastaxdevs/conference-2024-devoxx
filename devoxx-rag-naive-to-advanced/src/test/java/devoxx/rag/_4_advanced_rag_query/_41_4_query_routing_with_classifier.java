package devoxx.rag._4_advanced_rag_query;

import com.google.cloud.vertexai.api.Schema;
import com.google.cloud.vertexai.api.Type;
import dev.langchain4j.classification.EmbeddingModelTextClassifier;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
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
import java.util.Map;

import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _41_4_query_routing_with_classifier extends AbstractDevoxxTest {

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

        enum Category {
            DOG, HORSE
        }

        @Override
        public Collection<ContentRetriever> route(Query query) {
            VertexAiEmbeddingModel embeddingModel =
                VertexAiEmbeddingModel.builder()
                    .project(System.getenv("GCP_PROJECT_ID"))
                    .location(System.getenv("GCP_LOCATION"))
                    .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
                    .publisher("google")
                    .modelName("text-embedding-004")
                    .taskType(VertexAiEmbeddingModel.TaskType.CLASSIFICATION) // classification!
                    .build();


            var classifier =
                new EmbeddingModelTextClassifier<Category>(embeddingModel, Map.of(
                    Category.DOG, List.of(
                        "something about dogs", "dog, dogs, and puppies", "dog species"
                    ),
                    Category.HORSE, List.of(
                        "something about horses", "horse racing", "what kind of horse is it?"
                    )
                ));

            List<Category> category = classifier.classify(query.text());

            System.out.println(yellow("-> Category recognized: " + category));

            if (Category.HORSE.equals(category.get(0))) {
                return List.of(createRetriever("/text/johnny.txt"));
            } else if (Category.DOG.equals(category.get(0))) {
                return  List.of(createRetriever("/text/shadow.txt"));
            }

            return Collections.emptyList();
        }
    }
}