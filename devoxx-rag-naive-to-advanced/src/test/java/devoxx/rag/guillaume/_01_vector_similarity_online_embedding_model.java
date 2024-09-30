package devoxx.rag.guillaume;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

public class _01_vector_similarity_online_embedding_model {
    public static void main(String[] args) {

        var embeddingModel = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-multilingual-embedding-002")
            .publisher("google")
            .build();

        Response<Embedding> embeddingOne = embeddingModel.embed("young dog");
        Response<Embedding> embeddingTwo = embeddingModel.embed("jeune chien");

        double similarity = CosineSimilarity.between(embeddingOne.content(), embeddingTwo.content());

        System.out.println("similarity = " + similarity);

    }
}
