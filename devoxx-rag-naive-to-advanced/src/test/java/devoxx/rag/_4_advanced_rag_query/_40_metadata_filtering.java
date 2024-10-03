package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.exception.TooManyDocumentsToCountException;
import com.datastax.astra.client.model.Document;
import com.datastax.astra.client.model.Filter;
import com.datastax.astra.client.model.Filters;
import com.datastax.astra.client.model.FindIterable;
import com.datastax.astra.client.model.FindOptions;
import com.datastax.astra.client.model.Projection;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstracDevoxxSampleTest;
import devoxx.rag.Assistant;
import org.checkerframework.checker.units.qual.C;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static com.datastax.astra.client.model.Filters.eq;
import static com.datastax.astra.client.model.Filters.lt;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;
import static java.util.Arrays.asList;


public class _40_metadata_filtering extends AbstracDevoxxSampleTest {

    static final String COLLECTION_NAME = "quote";

    @Test
    public void should_show_metadata() throws TooManyDocumentsToCountException {
        System.out.println(yellow("Count documents"));
        System.out.println(getCollection(COLLECTION_NAME).countDocuments(1000));

        // List me all quotes from Aristotle and show me the quote and tags
        System.out.println(yellow("Show Aristotle quotes"));

        getCollection(COLLECTION_NAME).find(eq("authors", "aristotle"))
                .forEach(doc -> {
            System.out.println(doc.get("content"));
        });
    }

    @Test
    public void shouldRetrieveDocument() {
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(new AstraDbEmbeddingStore(getCollection(COLLECTION_NAME)))
                .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_TEXT))
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Enhance the content retriever to add meta data in the prompt
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                // Query Transformation
                .contentInjector(DefaultContentInjector
                        .builder()
                        .metadataKeysToInclude(asList("document_format",  "md5"))
                        .build())

                .build();

        // configuring it to use the components we've created above.
        Assistant ai = AiServices.builder(Assistant.class)
                //.contentRetriever(contentRetriever)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();


        String response = ai.answer("Who is Johnny?");
        System.out.println(response);
    }

}