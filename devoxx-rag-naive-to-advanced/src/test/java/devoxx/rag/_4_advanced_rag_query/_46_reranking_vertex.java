package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.model.vertexai.VertexAiScoringModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.stream.Collectors;

import static com.datastax.astra.internal.utils.AnsiUtils.*;

public class _46_reranking_vertex extends AbstractDevoxxTest {

    @Test
    public void shouldRerankResult() {

        // -----------------------
        // Ingesting document

        System.out.println(cyan("Ingesting document..."));

        Document document = loadDocumentText("text/berlin.txt");

        var embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);

        var splitter = new DocumentByParagraphSplitter(500, 50);

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(splitter)
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .build();
        ingestor.ingest(document);

        // -------------------
        // Retrieval

        System.out.println(cyan("Retrieving..."));

        // Re-Ranking
        ScoringModel scoringModel = VertexAiScoringModel.builder()
            .projectId(System.getenv("GCP_PROJECT_ID"))
            .projectNumber(System.getenv("GCP_PROJECT_NUM"))
            .location(System.getenv("GCP_LOCATION"))
            .model("semantic-ranker-default-004")
            .build();

        ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
            .scoringModel(scoringModel)
            .minScore(0.8)
            .build();

        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
            .embeddingStore(embeddingStore)
            .embeddingModel(embeddingModel)
            .maxResults(5)
            .minScore(0.7)
            .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
            .contentRetriever(retriever)
            .contentAggregator(contentAggregator)
            .build();

        interface Assistant {
            Result<String> answer(String query);
        }

        Assistant assistant = AiServices.builder(Assistant.class)
            .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_FLASH))
            .retrievalAugmentor(retrievalAugmentor)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .build();

        String query = "What's the population of Berlin?";
        System.out.println("\n" + green(query) + "\n");

        Result<String> result = assistant.answer(query);

        System.out.println(result.content());

        String sources = result.sources().stream()
            .map(content -> "-> " + content.textSegment().text())
            .collect(Collectors.joining("\n"));

        System.out.println(cyan("SOURCES:") + "\n");
        System.out.println(sources);


        // -----------------------
        // Plain search showing reranking score

        EmbeddingSearchResult<TextSegment> nonRankedSearchResults = embeddingStore.search(EmbeddingSearchRequest.builder()
            .maxResults(5)
            .minScore(0.7)
            .queryEmbedding(embeddingModel.embed(query).content())
            .build());

        String nonRankedSources = nonRankedSearchResults.matches().stream()
            .map(match ->
                "-> " +
                    yellow("(" + scoringModel.score(match.embedded(), query).content() + ") ") +
                    match.embedded().text())
            .collect(Collectors.joining("\n"));

        System.out.println("\n" + cyan("NON-RANKED SOURCES:") + "\n");
        System.out.println(nonRankedSources);
    }
}