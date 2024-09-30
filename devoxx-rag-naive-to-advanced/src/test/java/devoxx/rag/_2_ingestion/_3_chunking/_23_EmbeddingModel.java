package devoxx.rag._2_ingestion._3_chunking;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

public class _23_EmbeddingModel extends AbstracDevoxxSampleTest {


    @Test
    public void getEmbeddingFromGemini() {
        Response<Embedding> res =
        getEmbeddingModel(MODEL_EMBEDDING_MULTILINGUAL)
                .embed("HELLO this is a vector");

        System.out.println(res.content().dimension());
        System.out.println(res.content().vectorAsList());
    }


}
