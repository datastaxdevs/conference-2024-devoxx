package devoxx.demo._6_rag;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import devoxx.demo.utils.AbstractDevoxxTestSupport;
import devoxx.demo.utils.Assistant;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;
import static java.util.stream.Collectors.joining;


public class _62_NaiveRag_RetrievalTest extends AbstractDevoxxTestSupport {

    private static final Logger log = LoggerFactory.getLogger(_62_NaiveRag_RetrievalTest.class);

    @Test
    public void shouldRetrieveContent1() {

        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "Base your answer on the following information:\n"
                        + "{{rag-context}}");

        String question = "Who is Johnny?";

        // RAG CONTEXT
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = new AstraDbEmbeddingStore(getCollectionRAG())
                .search(EmbeddingSearchRequest.builder()
                        //.filter(metadataKey("document_format").isEqualTo("text"))
                        .queryEmbedding(getEmbeddingModelGecko().embed(question).content())
                        .minScore(0.5)
                        .maxResults(2)
                        .build()).matches();

        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("rag-context", relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(joining("\n\n")));
        log.info("{}", variables);

        Prompt prompt = promptTemplate.apply(variables);

        // See an answer from the model
        log.info(getChatLanguageModelChatBison().generate(prompt.toUserMessage()).content().text());
    }

    @Test
    public void shouldRetrieveContent2() {

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(new AstraDbEmbeddingStore(getCollectionRAG()))
                .embeddingModel(getEmbeddingModelGecko())
                .maxResults(2)
                .minScore(0.5)
                .build();

        // configuring it to use the components we've created above.
        Assistant ai = AiServices.builder(Assistant.class)
                .contentRetriever(contentRetriever)
                .chatLanguageModel(getChatLanguageModelChatBison())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        String response = ai.answer("Who is Johnny?");
        System.out.println(response);
    }

}