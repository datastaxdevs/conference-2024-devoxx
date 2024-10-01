package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _22_vectors extends AbstracDevoxxSampleTest  {

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
    public void should_compute_vector_similarity() {

    }

}
