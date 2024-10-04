package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

public class _33_3_embedding_retrieval_task {
    public static void main(String[] args) {
        var embeddingForDocs = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .taskType(VertexAiEmbeddingModel.TaskType.RETRIEVAL_DOCUMENT)
            .build();

        var definitionText = """
            Embedding models are machine learning models that convert complex data, like text or images, 
            into numerical representations called embeddings. These embeddings capture the relationships 
            between different pieces of data, allowing machines to understand and process them more 
            effectively. They are used in various applications, including natural language processing, 
            image and video analysis, and recommendation systems.""";

        Response<Embedding> definitionEmbedding = embeddingForDocs.embed(definitionText);

        var embeddingForQuery = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .taskType(VertexAiEmbeddingModel.TaskType.RETRIEVAL_QUERY)
            .build();

        var questionText = "What are embedding models?";

        Response<Embedding> questionEmbedding = embeddingForQuery.embed(questionText);

        double similarity = CosineSimilarity.between(
            definitionEmbedding.content(),
            questionEmbedding.content()
        );

        System.out.println("similarity = " + similarity);
    }
}
