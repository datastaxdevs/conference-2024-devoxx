package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.similarity.CosineSimilarity;
import devoxx.rag.similarity.DotProductSimilarity;
import devoxx.rag.similarity.EuclideanSimilarity;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.dtsx.astra.sdk.utils.observability.AnsiUtils.yellow;

public class _32_vectors_similarity extends AbstractDevoxxTest {

    @Test
    public void should_show_different_similarities() {

        var embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        Embedding one = embeddingModel.embed("baby dog").content();
        Embedding two = embeddingModel.embed("jeune chien").content();

        // Cosine Similarity
        System.out.println(yellow("AllMiniLmL6V2"));
        System.out.println(cyan("     Cosine Similarity: ")
                + new CosineSimilarity().compute(one, two));
        System.out.println(cyan("  Euclidean Similarity: ")
                + new EuclideanSimilarity().compute(one, two));
        System.out.println(cyan(" DotProduct Similarity: ")
                + new DotProductSimilarity().compute(one, two));

        // Not multilingual
        var embeddingModel2 = getEmbeddingModel(MODEL_EMBEDDING_MULTILINGUAL);
        Embedding three = embeddingModel2.embed("baby dog").content();
        Embedding four = embeddingModel2.embed("jeune chien").content();

        // Cosine Similarity
        System.out.println(yellow("\nVertexAI " + MODEL_EMBEDDING_MULTILINGUAL));
        System.out.println(cyan("     Cosine Similarity: ")
                + new CosineSimilarity().compute(three, four));
        System.out.println(cyan("  Euclidean Similarity: ")
                + new EuclideanSimilarity().compute(three, four));
        System.out.println(cyan(" DotProduct Similarity: ")
                + new DotProductSimilarity().compute(three, four));
    }

}
