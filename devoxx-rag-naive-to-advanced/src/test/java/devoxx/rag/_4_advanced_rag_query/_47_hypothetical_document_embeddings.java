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
        ingestor.ingest(Document.from(text));

        String queryString = "What are the main dimensionality reduction techniques?";

        System.out.println(cyan("\n———————— QUESTION —————————————————————————————————————\n"));
        System.out.println(queryString);

        Response<AiMessage> hypotheticalAnswer = chatModel.generate(
            List.of(
                SystemMessage.from("Provide a short and concise answer to the user's questions"),
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