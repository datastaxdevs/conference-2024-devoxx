package devoxx.rag.experiments;

import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiScoringModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.util.List;

public class _60_intuition_scoring_embedding {
    public static void main(String[] args) {
        String text = """
            Chapter 6: Towing, Cargo, and Luggage
            
            Towing
            
            Your Cymbal Starlight 2024 is not equipped to tow a trailer.
            
            Cargo
            
            The Cymbal Starlight 2024 has a cargo capacity of 13.5 cubic feet. The cargo area is located in the trunk of
            the vehicle.
            
            To access the cargo area, open the trunk lid using the trunk release lever located in the driver's footwell.
            
            When loading cargo into the trunk, be sure to distribute the weight evenly. Do not overload the trunk, as this
            could affect the vehicle's handling and stability.
            """;

        DocumentBySentenceSplitter splitter = new DocumentBySentenceSplitter(300, 50);
        String[] chunks = splitter.split(text);

        for (String chunk : chunks) {
            System.out.println("\n---\n" + chunk);
        }

        var scoringModel = VertexAiScoringModel.builder()
            .projectId(System.getenv("GCP_PROJECT_ID"))
            .projectNumber(System.getenv("GCP_PROJECT_NUM"))
            .location(System.getenv("GCP_LOCATION"))
            .model("semantic-ranker-512")
            .build();

        String query = "What is the cargo capacity of Cymbal Starlight?";

        List<TextSegment> segments = List.of(
            TextSegment.from("The Cymbal Starlight 2024 has a cargo capacity of 13.5 cubic feet."),
            TextSegment.from("""
                Cargo
                
                The Cymbal Starlight 2024 has a cargo capacity of 13.5 cubic feet.
                The cargo area is located in the trunk of the vehicle.
                """),
            TextSegment.from(text)
        );

        System.out.println(scoringModel.scoreAll(segments, query));

        var embeddingModel = VertexAiEmbeddingModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .build();

        Embedding embeddedQuery = embeddingModel.embed(query).content();
        List<Embedding> embeddedSegments = embeddingModel.embedAll(segments).content();

        for (Embedding embeddedSegment : embeddedSegments) {
            System.out.println(CosineSimilarity.between(embeddedQuery, embeddedSegment));
        }

    }
}