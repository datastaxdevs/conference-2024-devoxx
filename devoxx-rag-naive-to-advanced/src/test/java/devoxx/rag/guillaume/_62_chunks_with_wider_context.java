package devoxx.rag.guillaume;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class _62_chunks_with_wider_context {
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

//        DocumentSplitter splitter = DocumentSplitters.recursive(100, 20);
        DocumentSplitter splitter = new DocumentBySentenceSplitter(100, 20);
        List<TextSegment> textSegments = splitter.split(Document.from(text));

        for (TextSegment textSegment : textSegments) {
            System.out.println("\n---\n" + textSegment.text());
        }

        System.out.println("\n------------ parent-child relationship ------------");

        List<List<TextSegment>> regroupedSegments = slidingWindow(textSegments, 1, 2);

        List<TextSegment> fusedGroupedSegments = regroupedSegments.stream()
            .map(textSegmentsList -> TextSegment.from(textSegmentsList.stream()
                .map(TextSegment::text)
                .collect(Collectors.joining("\n"))))
            .toList();

        for (TextSegment oneGroupedSegment : fusedGroupedSegments) {
            System.out.println("\n---\n" + oneGroupedSegment.text());
        }

        Map<TextSegment, String> childParentRelationship = new HashMap<>();
        for (int i = 0; i < textSegments.size(); i++) {
            TextSegment textSegment = textSegments.get(i);
            TextSegment fusedParentSegment = fusedGroupedSegments.get(i);
            childParentRelationship.put(textSegment, fusedParentSegment.text());
        }

        // ======================

        var embeddingModel = VertexAiEmbeddingModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .publisher("google")
            .modelName("text-embedding-004")
            .build();

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(splitter)
//            .documentTransformer(document -> document)
//            .textSegmentTransformer(textSegment -> textSegment)
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .build();

        ingestor.ingest(Document.from(text));

        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .maxResults(4)
            .minScore(0.7)
            .build();

        List<Content> retrieved = retriever.retrieve(Query.from("What is the cargo capacity of Cymbal Starlight?"));

        for (Content content : retrieved) {
            System.out.println("\n===\n" + content);
            System.out.println("\n--- Parent ----\n" + childParentRelationship.get(content.textSegment()));
        }
    }

    private static <T> List<List<T>> slidingWindow(List<T> input, int nbBefore, int nbAfter) {
        return IntStream.range(0, input.size())
            .mapToObj(i -> IntStream.rangeClosed(i - nbBefore, i + nbAfter)
                .filter(j -> j >= 0 && j < input.size())
                .mapToObj(input::get)
                .collect(Collectors.toList()))
            .collect(Collectors.toList());
    }

    static class ParentChildEmbeddingStoreIngestor {

    }
}