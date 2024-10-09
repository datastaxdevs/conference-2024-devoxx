package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

public class _41_2_query_routing_websearch extends AbstractDevoxxTest {

    @Test
    public void testQueryRouting() {

        // Let's create our web search content retriever.
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
            .apiKey(System.getenv("TAVILY_API_KEY"))
            .build();

        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
            .webSearchEngine(webSearchEngine)
            .maxResults(3)
            .build();

        // Let's create a query router that will route each query to both retrievers.
        QueryRouter queryRouter = new DefaultQueryRouter(createRetriever("/text/berlin.txt"),
            webSearchContentRetriever);

        // Let's define our retrieval augmentor for advanced RAG
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
            .queryRouter(queryRouter)
            .build();

        Assistant assistant = AiServices.builder(Assistant.class)
            .retrievalAugmentor(retrievalAugmentor)
            .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
            .build();

        System.out.println(assistant.answer("What is the current population of Paris?"));
    }
}