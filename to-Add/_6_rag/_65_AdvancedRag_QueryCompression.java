package devoxx.rag._6_rag;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import devoxx.rag.TestSupport;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class _65_AdvancedRag_QueryCompression extends TestSupport {


    @Test
    public void shouldTestQueryCompression() {

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                // Commons Retriever
                .contentRetriever(createRetriever("/johnny.txt"))
                // Add a Query Transformation
                .queryTransformer(new CompressingQueryTransformer(getChatLanguageModelChatBison()))
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(getChatLanguageModelChatBison())
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        System.out.println(assistant.answer("Give me the name of the horse"));
        System.out.println(assistant.answer("Can you tell where he lives ?"));
        System.out.println(assistant.answer("What does he do ?"));

    }

}