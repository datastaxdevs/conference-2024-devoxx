package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import java.util.List;

public class _43_query_compression extends AbstractDevoxxTest {


    @Test
    public void shouldTestQueryCompression() {

        // Let's create our web search content retriever.

        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(TavilyWebSearchEngine.builder().apiKey(System.getenv("TAVILY_API_KEY")).build())
                .maxResults(3)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new DefaultQueryRouter(createRetriever("/text/berlin.txt"),webSearchContentRetriever))
                .queryTransformer(new CompressingQueryTransformer(getChatLanguageModel(MODEL_GEMINI_PRO)))
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        Response<AiMessage> hypotheticalAnswer = getChatLanguageModel(MODEL_GEMINI_PRO).generate(
                List.of(
                        SystemMessage.from("Provide answer to the question to the best knowledge"),
                        UserMessage.from("What is the population of Berlin ?")
                )
        );
        System.out.println(assistant.answer("What is the population of Berlin ?"));

    }

}