package devoxx.rag._6_rag;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import devoxx.rag.TestSupport;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static java.util.Arrays.asList;
import static java.util.stream.Collectors.joining;


public class _63_AdvancedRag_AugmentWithMetadata extends TestSupport {

    @Test
    public void shouldRetrieveDocument() {

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(new AstraDbEmbeddingStore(getCollectionRAG()))
                .embeddingModel(getEmbeddingModelGecko())
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
                .chatLanguageModel(getChatLanguageModelChatBison())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();


        String response = ai.answer("Who is Johnny?");
        System.out.println(response);
    }

}