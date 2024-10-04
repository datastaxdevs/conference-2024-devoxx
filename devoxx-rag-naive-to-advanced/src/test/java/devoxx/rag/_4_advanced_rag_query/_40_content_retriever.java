package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _40_content_retriever extends AbstractDevoxxTest  {

    @Test
    public void sampleRetriever() {

        ContentRetriever retriever = createRetriever("/text/johnny.txt");

        Assistant assistant = AiServices.builder(Assistant.class)
                .contentRetriever(retriever)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
                .build();

        System.out.println(yellow("Chunks"));
        retriever.retrieve(Query.from("What is the name of the horse"))
                .forEach(content -> System.out.println("-" + content.textSegment().text()));

        System.out.println(yellow("\nResult"));
        System.out.println(assistant.answer("Give me the name of the horse"));
    }
}
