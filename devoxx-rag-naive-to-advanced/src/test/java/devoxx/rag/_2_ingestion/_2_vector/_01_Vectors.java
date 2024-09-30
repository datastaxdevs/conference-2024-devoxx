package devoxx.rag._2_ingestion._2_vector;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

public class _01_Vectors extends AbstracDevoxxSampleTest  {

    @Test
    public void should_vector_normalized() {
        Response<Embedding> res = getEmbeddingModel(MODEL_EMBEDDING_TEXT)
                .embed("HELLO this is a vector");
        float sum = 0;
        for (float value : res.content().vectorAsList()) {
            sum += value * value;
        }
        System.out.println("L2 Norm: " + (float) Math.sqrt(sum));
    }


    @Test
    public void testVectors() {
            
    }

}
