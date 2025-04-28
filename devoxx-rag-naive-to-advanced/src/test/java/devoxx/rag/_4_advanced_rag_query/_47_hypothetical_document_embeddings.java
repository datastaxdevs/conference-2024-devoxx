package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _47_hypothetical_document_embeddings extends AbstractDevoxxTest {

    @Test
    public void hydetheticalDocumentEmbeddings() {
        Document document = loadDocumentText("text/berlin.txt");

        var chatModel = getChatLanguageModel(MODEL_GEMINI_PRO);
        var embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);

        DocumentByParagraphSplitter splitter = new DocumentByParagraphSplitter(1000, 0);

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(splitter)
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .build();
        ingestor.ingest(document);

        String queryString = "What's the population of Berlin?'";

        System.out.println(cyan("\n———————— QUESTION —————————————————————————————————————\n"));
        System.out.println(queryString);

        Response<AiMessage> hypotheticalAnswer = chatModel.generate(
            List.of(
                SystemMessage.from(
                    "Answer concisely to user's questions with a clear and explicit subject, verb, and object"),
                UserMessage.from(queryString)
            )
        );
        System.out.println(cyan("\n———————— HYPOTHETICAL ANSWER ——————————————————————————\n"));
        String hypotheticalQuestion = hypotheticalAnswer.content().text();
        System.out.println(hypotheticalQuestion);

        // Searching...

        EmbeddingSearchResult<TextSegment> searchDirect = embeddingStore.search(EmbeddingSearchRequest.builder()
            .minScore(0.7)
            .maxResults(5)
            .queryEmbedding(embeddingModel.embed(queryString).content())
            .build());

        EmbeddingSearchResult<TextSegment> searchHypothetical = embeddingStore.search(EmbeddingSearchRequest.builder()
            .minScore(0.7)
            .maxResults(5)
            .queryEmbedding(embeddingModel.embed(hypotheticalQuestion).content())
            .build());

        System.out.println(cyan("\n———————— RESULT FROM QUERY ————————————————————————————\n"));
        searchDirect.matches().forEach(match -> {
            System.out.println(yellow("\n-> Similarity: " + match.score() +  " ————————\n") + match.embedded().text());
        });

        System.out.println(cyan("\n———————— RESULT FROM HYPOTHETICAL ANSWER ——————————————\n"));
        searchHypothetical.matches().forEach(match -> {
            System.out.println(yellow("\n-> Similarity: " + match.score() +  " ————————\n") + match.embedded().text());
        });
    }
}