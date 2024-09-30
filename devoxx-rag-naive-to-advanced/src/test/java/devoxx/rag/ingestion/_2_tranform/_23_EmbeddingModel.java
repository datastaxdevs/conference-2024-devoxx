package devoxx.rag.ingestion._2_tranform;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

public class _23_EmbeddingModel extends AbstracDevoxxSampleTest {


    @Test
    public void should_vector_normalized() {
        Response<Embedding> res = getEmbeddingModel(MODEL_EMBEDDING_GECKO).embed("HELLO this is a vector");

        // Compute L2
        float sum = 0;
        for (float value : res.content().vectorAsList()) {
            sum += value * value;
        }
        float L2 =  (float) Math.sqrt(sum);
        System.out.println("L2 Norm: " + L2);

        // Compute L1
        sum = 0;
        for (float value : res.content().vectorAsList()) {
            sum += Math.abs(value);
        }
        System.out.println("L1 Norm: " + sum);
    }


    @Test
    public void getEmbeddingFromGemini() {
        Response<Embedding> res =
        getEmbeddingModel(MODEL_EMBEDDING_GECKO).embed("HELLO this is a vector");

        System.out.println(res.content().dimension());
        System.out.println(res.content().vectorAsList());


    }
}
