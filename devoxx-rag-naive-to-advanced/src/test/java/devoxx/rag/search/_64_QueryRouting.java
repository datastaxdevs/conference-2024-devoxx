package devoxx.rag.search;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstracDevoxxSampleTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.List;

public class _64_QueryRouting extends AbstracDevoxxSampleTest {


    @Test
    public void testQueryRouting() {
        // Our guy for advanced RAG
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

    private static class MyRouter extends AbstracDevoxxSampleTest implements QueryRouter  {

        @Override
        public Collection<ContentRetriever> route(Query query) {
            if (query.text().contains("horse")) {
                return List.of(createRetriever("/johnny.txt"));
            } else if (query.text().contains("dog")) {
                return  List.of(createRetriever("/shadow.txt"));
            }
            return List.of();
        }

    }
}