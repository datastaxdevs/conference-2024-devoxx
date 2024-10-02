package devoxx.rag.experiments;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.CosineSimilarity;

public class _00_vector_similarity_local_embedding_model {
    public static void main(String[] args) {

        var embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

        Response<Embedding> embeddingOne = embeddingModel.embed("baby dog");
        Response<Embedding> embeddingTwo = embeddingModel.embed("puppy");

        double similarity = CosineSimilarity.between(embeddingOne.content(), embeddingTwo.content());

        System.out.println("similarity = " + similarity);

    }
}
