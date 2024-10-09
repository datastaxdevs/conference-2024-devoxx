package devoxx.rag._4_advanced_rag_query;

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
import java.util.List;

public class _41_1_query_routing_simple extends AbstractDevoxxTest {

    @Test
    public void testQueryRouting() {
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new MyRouter())
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
    private static class MyRouter extends AbstractDevoxxTest implements QueryRouter  {

        @Override
        public Collection<ContentRetriever> route(Query query) {
            if (query.text().contains("horse")) {
                return List.of(createRetriever("/text/johnny.txt"));
            } else if (query.text().contains("dog")) {
                return  List.of(createRetriever("/text/shadow.txt"));
            }
            return List.of();
        }
    }
}