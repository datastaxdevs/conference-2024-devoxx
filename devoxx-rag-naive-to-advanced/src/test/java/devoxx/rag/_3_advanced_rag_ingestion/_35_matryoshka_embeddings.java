package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;

public class _35_matryoshka_embeddings {
    public static void main(String[] args) {

        var embeddingModel768 = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .outputDimensionality(768)
            .build();

        var embeddingModel256 = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .outputDimensionality(256)
            .build();

        Response<Embedding> embeddingOne768 = embeddingModel768.embed("young dog");
        Response<Embedding> embeddingTwo256 = embeddingModel256.embed("young dog");

        System.out.println("embeddingOne768 = " + embeddingOne768.content().vectorAsList());
        System.out.println("embeddingTwo256 = " + embeddingTwo256.content().vectorAsList());

        System.out.println(embeddingTwo256.content().vectorAsList().equals(
            embeddingOne768.content().vectorAsList().subList(0, 256)));
    }
}
