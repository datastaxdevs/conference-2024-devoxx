package devoxx.rag.experiments;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class _10_in_memory_ingestion {
    public static void main(String[] args) {

        VertexAiEmbeddingModel embeddingModel = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .build();

        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        List.of(
            "I like football.",
            "The weather is good today.",
            "American football is not really a foot ball sport.",
            "Cats are my favorite animals.",
            "Devoxx Belgium takes place in Antwerp."
        ).forEach(text -> {
            TextSegment segment = TextSegment.from(text);
            Embedding embedding = embeddingModel.embed(segment).content();
            embeddingStore.add(embedding, segment);
        });

        Embedding queryEmbedding = embeddingModel.embed("What is your preferred sport?").content();

        EmbeddingSearchResult<TextSegment> relevant = embeddingStore.search(
            EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(4)
                .minScore(0.7)
                .build()
        );

        relevant.matches().forEach(embeddingMatch -> {
            System.out.println("[" + embeddingMatch.score() + "] --> " + embeddingMatch.embedded().text());
        });
    }
}
