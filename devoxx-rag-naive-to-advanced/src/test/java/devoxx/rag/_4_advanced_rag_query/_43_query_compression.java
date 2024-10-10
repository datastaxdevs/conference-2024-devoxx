package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static devoxx.rag._3_advanced_rag_ingestion._37_hypothetical_questions_embedding.getEmbeddingStore;

public class _43_query_compression extends AbstractDevoxxTest {

    @Test
    public void shouldTestQueryCompression() {

//        ContentRetriever webSearchContentRetriever = WebSearchContentRetriever.builder()
//                .webSearchEngine(TavilyWebSearchEngine.builder().apiKey(System.getenv("TAVILY_API_KEY")).build())
//                .maxResults(3)
//                .build();

        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
            .embeddingStore(getEmbeddingStore())
            .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_TEXT))
            .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
//                .queryRouter(new DefaultQueryRouter(retriever, webSearchContentRetriever))
                .contentRetriever(retriever)
                .queryTransformer(new CompressingQueryTransformer(getChatLanguageModel(MODEL_GEMINI_FLASH)))
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