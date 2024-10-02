package devoxx.rag.experiments;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Map;

public class _66_contextual_retrieval {
    public static void main(String[] args) {
        String text = """
            Dimensionality Reduction: Simplifying Complex Data
            
            Dimensionality reduction is a technique used to simplify complex datasets by reducing the number of features (or dimensions) while preserving as much relevant information as possible. 
            This process can be beneficial for several reasons:
            
            Computational Efficiency: High-dimensional data can be computationally expensive to process. By reducing dimensionality, you can significantly speed up algorithms and improve model performance.
            Visualization: Visualizing high-dimensional data is challenging. Dimensionality reduction techniques can help to create more interpretable visualizations, making it easier to understand relationships and patterns within the data.
            Noise Reduction: Redundant or irrelevant features can introduce noise into your data. Dimensionality reduction can help to eliminate or reduce the impact of noise.
            Feature Engineering: By combining or transforming features, dimensionality reduction can create new, more informative features that can improve model performance.
            
            Common Dimensionality Reduction Techniques
            
            Principal Component Analysis (PCA): This is a popular technique that finds a new set of uncorrelated variables (principal components) that capture the most variance in the data.
            Linear Discriminant Analysis (LDA): LDA is a supervised technique that seeks to find a projection that maximizes the separation between classes.
            t-SNE: t-SNE is a nonlinear technique that is particularly effective for visualizing high-dimensional data in low-dimensional spaces.
            Autoencoders: Autoencoders are neural networks that learn to compress and reconstruct data. They can be used for dimensionality reduction by training them to learn a lower-dimensional representation of the input data.
            
            When to Use Dimensionality Reduction:
            
            High-dimensional data: When you have a large number of features.
            Computational efficiency: When you need to improve the speed of your algorithms.
            Visualization: When you want to visualize complex data.
            Noise reduction: When you suspect your data contains noise.
            Feature engineering: When you want to create new, more informative features.
            
            Choosing the Right Technique:
            
            The best dimensionality reduction technique depends on your specific use case and the characteristics of your data. Consider factors such as the number of dimensions, the nature of the data (linear vs. nonlinear), and whether you have labeled data.
            
            By understanding dimensionality reduction and applying the appropriate techniques, you can simplify complex datasets, improve model performance, and gain valuable insights from your data.
            """;

        var gemini = VertexAiGeminiChatModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("gemini-1.5-flash-002")
            .build();

        var embeddingModel = VertexAiEmbeddingModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .build();

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        PromptTemplate promptTemplate = PromptTemplate.from("""
            <document>
            {{wholeDocument}}
            </document>
            Here is the chunk we want to situate within the whole document
            <chunk>
            {{chunk}}
            </chunk>
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. \
            Answer only with the succinct context and nothing else.
            """);

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(DocumentSplitters.recursive(1000, 0))
            .embeddingStore(embeddingStore)
            .embeddingModel(embeddingModel)
            .textSegmentTransformer(segment -> {
                Response<AiMessage> generatedChunk = gemini.generate(promptTemplate.apply(Map.of(
                        "chunk", segment.text(),
                        "wholeDocument", text))
                    .toUserMessage());

                System.out.println("\n=======================================================");
                System.out.println("ORIGINAL:\n\n" + segment.text());
                System.out.println("\n----------------");
                System.out.println("TRANSFORMED:\n\n" + generatedChunk.content().text());

                return TextSegment.from(generatedChunk.content().text(), new Metadata().put("original", segment.text()));
            })
            .build();
        ingestor.ingest(Document.from(text));

        String question = "What are the main dimensionality reduction techniques?";

        EmbeddingSearchResult<TextSegment> results = embeddingStore.search(EmbeddingSearchRequest.builder()
            .minScore(0.7)
            .maxResults(5)
            .queryEmbedding(embeddingModel.embed(question).content())
            .build());

        results.matches().forEach(match -> {
            System.out.println("\n--- " + match.score() +  " -----\n");
            System.out.println("ORIGINAL:\n\n" + match.embedded().metadata().getString("original"));
            System.out.println("\nTRANSFORMED:\n\n" + match.embedded().text());

        });

    }
}