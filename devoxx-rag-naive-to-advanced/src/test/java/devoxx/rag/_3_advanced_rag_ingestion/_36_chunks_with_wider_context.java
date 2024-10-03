package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.datastax.astra.internal.utils.AnsiUtils.*;

public class _36_chunks_with_wider_context extends AbstractDevoxxTest {

    public static final String PARENT_CONTEXT_KEY = "parentContext";

    @Test
    public void slidingWindowOfSentences() {
        Document documentAboutBerlin = loadDocumentText("text/berlin.txt");

        DocumentSplitter splitter = new DocumentBySentenceSplitter(200, 20);
        List<TextSegment> textSegments = splitter.split(documentAboutBerlin);

        System.out.println(cyan("SENTENCES"));
        for (TextSegment textSegment : textSegments) {
            System.out.println("-> " + textSegment.text());
        }

        // ======================================================
        // Embed the sentences, but store the surrounding context

        List<ParentChild> parentChildren = slidingWindow(textSegments, 1, 2);

        var embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        List<TextSegment> embeddedSegments = parentChildren.stream()
            .map(ParentChild::embeddedChild)
            .toList();
        List<Embedding> embeddings = embeddingModel.embedAll(embeddedSegments).content();
        embeddingStore.addAll(embeddings, embeddedSegments);

        // =================================================
        // Search the vector store for the embedded sentence

        String queryString = "How many inhabitants live in Berlin?";

        System.out.println(magenta("\nQUESTION: ") + queryString);

        EmbeddingSearchResult<TextSegment> searchResults = embeddingStore.search(EmbeddingSearchRequest.builder()
            .maxResults(4)
            .minScore(0.7)
            .queryEmbedding(embeddingModel.embed(queryString).content())
            .build());

        searchResults.matches().forEach(match -> {
            System.out.println(yellow("\n-> Score: " + match.score()) +
                "\n" + cyan("Embedded child: ") + match.embedded().text() +
                "\n" + cyan("Parent context: ") + match.embedded().metadata().getString(PARENT_CONTEXT_KEY));
        });

        // =================================
        // Ask Gemini to generate a response

        ChatLanguageModel chatModel = getChatLanguageModel(MODEL_GEMINI_FLASH);

        Response<AiMessage> response = chatModel.generate(PromptTemplate.from("""
            You must answer the following question:
            
            {{question}}
            
            Base your answer on the following documentation extracts:

            {{extracts}}
            """).apply(Map.of(
            "question", queryString,
            "extracts", searchResults.matches().stream()
                .map(match -> match.embedded().metadata().getString(PARENT_CONTEXT_KEY))
                .collect(Collectors.joining("\n---\n", "\n---\n", "\n---\n"))
        )).toUserMessage());

        System.out.println(magenta("\nRESPONSE:\n") + response.content().text());
    }

    record ParentChild(
        String parentContext,
        TextSegment embeddedChild
    ) {
    }

    private static List<ParentChild> slidingWindow(List<TextSegment> input, int nbBefore, int nbAfter) {
        return IntStream.range(0, input.size())
            .mapToObj(i -> {
                TextSegment textSegment = input.get(i);
                String parent = IntStream.rangeClosed(i - nbBefore, i + nbAfter)
                    .filter(j -> j >= 0 && j < input.size())
                    .mapToObj(j -> input.get(j).text())
                    .collect(Collectors.joining(" "));
                textSegment.metadata().put(PARENT_CONTEXT_KEY, parent);
                return new ParentChild(parent, textSegment);
            })
            .collect(Collectors.toList());
    }
}