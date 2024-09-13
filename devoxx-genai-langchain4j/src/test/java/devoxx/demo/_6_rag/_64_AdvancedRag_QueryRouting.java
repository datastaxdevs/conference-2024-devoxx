package devoxx.demo._6_rag;

import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.demo.utils.AbstractDevoxxTestSupport;
import devoxx.demo.utils.Assistant;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class _64_AdvancedRag_QueryRouting  extends AbstractDevoxxTestSupport {


    @Test
    public void testQueryRouting() {
        // Our guy for advanced RAG
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new MyRouter())
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModelChatBison())
                .build();

        System.out.println(assistant.answer("Give me the name of the horse"));
        System.out.println(assistant.answer("Give me the name of the dog"));

    }

    private static class MyRouter extends AbstractDevoxxTestSupport implements QueryRouter  {

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