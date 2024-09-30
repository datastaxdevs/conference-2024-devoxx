package devoxx.rag._3_search._2_VectorSearch;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstracDevoxxSampleTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static java.util.Arrays.asList;


public class _63_Filter_MetaDataFiltering extends AbstracDevoxxSampleTest {

    @Test
    public void shouldRetrieveDocument() {

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(new AstraDbEmbeddingStore(getCollection("demo")))
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