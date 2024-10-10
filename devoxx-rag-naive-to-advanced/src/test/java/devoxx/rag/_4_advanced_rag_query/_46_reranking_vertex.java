package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.cohere.CohereScoringModel;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.model.vertexai.VertexAiScoringModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static devoxx.rag._3_advanced_rag_ingestion._37_hypothetical_questions_embedding.getEmbeddingStore;

public class _46_reranking_vertex extends AbstractDevoxxTest {

    @Test
    public void shouldRerankResult() {

        // -----------------------
        // Ingesting document

        System.out.println(cyan("Ingesting document..."));

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

        // -------------------
        // Retrieval

        System.out.println(cyan("Retrieving..."));

        // Re Ranking
        ScoringModel scoringModel = VertexAiScoringModel.builder()
            .projectId(System.getenv("GCP_PROJECT_ID"))
            .projectNumber(System.getenv("GCP_PROJECT_NUM"))
            .location(System.getenv("GCP_LOCATION"))
            .model("semantic-ranker-512")
            .build();

        ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .minScore(0.8)
                .build();

        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
            .embeddingStore(embeddingStore)
            .embeddingModel(embeddingModel)
            .maxResults(10)
            .minScore(0.7)
            .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(retriever)
                .contentAggregator(contentAggregator)
                .build();

        interface Assistant {
            String answer(String query);
        }

        Assistant assistant =  AiServices.builder(Assistant.class)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_FLASH))
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        System.out.println( assistant.answer("What are the main dimensionality reduction techniques?"));
    }
}