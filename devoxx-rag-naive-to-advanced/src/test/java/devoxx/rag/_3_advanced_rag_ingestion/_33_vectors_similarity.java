package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.CosineSimilarity;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _33_vectors_similarity extends AbstracDevoxxSampleTest  {

    /**
     * Euclidean
     */
    public static double euclideanDistance(List<Float> vector1, List<Float> vector2) {
        double sum = 0.0;
        for (int i = 0; i < vector1.size(); i++) {
            sum += Math.pow(vector1.get(i) - vector2.get(i), 2);
        }
        return Math.sqrt(sum);
    }

    /**
     * DotProduct
     */
    public static double dotProductSimilarity(List<Float> vector1, List<Float> vector2) {
        double dotProduct = 0.0;
        for (int i = 0; i < vector1.size(); i++) {
            dotProduct += vector1.get(i) * vector2.get(i);
        }
        return dotProduct;
    }

    @Test
    public void should_show_different_similarities() {
        // Not multilingual
        var embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        Response<Embedding> embeddingOne = embeddingModel.embed("baby dog");
        Response<Embedding> embeddingTwo = embeddingModel.embed("puppy");

        // Cosine Similarity
        System.out.println(cyan("    Cosine Similarity: ") +
                CosineSimilarity.between(embeddingOne.content(), embeddingTwo.content()));

        System.out.println(cyan(" Euclidean Similarity: ") +
                euclideanDistance(embeddingOne.content().vectorAsList(),
                        embeddingTwo.content().vectorAsList()));

        System.out.println(cyan("DotProduct Similarity: ") +
                dotProductSimilarity(embeddingOne.content().vectorAsList(),
                        embeddingTwo.content().vectorAsList()));
    }


    @Test
    public void should_show_multilingual() {
        // Not multilingual
        var embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_MULTILINGUAL);
        Response<Embedding> embeddingOne = embeddingModel.embed("baby dog");
        Response<Embedding> embeddingTwo = embeddingModel.embed("puppy");

        // Cosine Similarity
        System.out.println(cyan("    Cosine Similarity: ") +
                CosineSimilarity.between(embeddingOne.content(), embeddingTwo.content()));

        System.out.println(cyan(" Euclidean Similarity: ") +
                euclideanDistance(embeddingOne.content().vectorAsList(),
                        embeddingTwo.content().vectorAsList()));

        System.out.println(cyan("DotProduct Similarity: ") +
                dotProductSimilarity(embeddingOne.content().vectorAsList(),
                        embeddingTwo.content().vectorAsList()));
    }


}
