package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _32_vectors extends AbstracDevoxxSampleTest  {

    @Test
    public void should_vector_normalized() {
        String chunk = "HELLO this is a vector";
        Response<Embedding> res = getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed(chunk);

        // The Model has a dimensionality o 768
        System.out.println(cyan("Vector: ") + res.content().vectorAsList());
        System.out.println(cyan("Dimensionality: ") + res.content().dimension());

        float sum = 0;
        for (float value : res.content().vectorAsList()) {
            sum += value * value;
        }
        System.out.println(cyan("L2 Norm: ") + (float) Math.sqrt(sum));
    }

    @Test
    public void should_compare_vectors() {

        String chunk = "HELLO this is a vector";
        Response<Embedding> chunk1     = getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed(chunk);
        Response<Embedding> multimodal = getEmbeddingModel(MODEL_EMBEDDING_MULTILINGUAL).embed(chunk);

        // Compute L2 Distance
        double sum = 0.0;
        for (int i = 0; i < chunk1.content().vectorAsList().size(); i++) {
            sum += Math.pow(chunk1.content().vectorAsList().get(i) - multimodal.content().vectorAsList().get(i), 2);
        }
        double distance =  Math.sqrt(sum);

        // Same input but different models the distance in vector is pretty high, keep the same embedding model
        System.out.println(cyan("L2 Distance: ") + distance);
    }


}
