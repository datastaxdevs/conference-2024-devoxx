package devoxx.rag._1_introduction;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _12_embedding_model extends AbstracDevoxxSampleTest {

    @Test
    public void should_illustrate_embedding_model() {
        String chunk = "HELLO this is a vector";
        EmbeddingModel embeddingModel = VertexAiEmbeddingModel.builder()
                .project(System.getenv("GCP_PROJECT_ID"))
                .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
                .location(System.getenv("GCP_LOCATION"))
                .publisher("google")
                .modelName("text-embedding-004")
                .build();

        Response<Embedding> res = embeddingModel.embed(chunk);

        // The Model has a dimensionality o 768
        System.out.println(cyan("Vector: ") + res.content().vectorAsList());
        System.out.println(cyan("Dimensionality: ") + res.content().dimension());

    }
}
