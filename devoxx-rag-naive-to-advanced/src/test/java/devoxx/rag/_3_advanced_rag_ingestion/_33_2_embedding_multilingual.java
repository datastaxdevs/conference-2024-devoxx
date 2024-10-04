package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.CosineSimilarity;
import org.junit.jupiter.api.Test;

public class _33_2_embedding_multilingual {

    @Test
    public void should_use_local_embedding_model() {
        var embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        Response<Embedding> embeddingOne = embeddingModel.embed("baby dog");
        Response<Embedding> embeddingTwo = embeddingModel.embed("puppy");
        double similarity = CosineSimilarity.between(embeddingOne.content(), embeddingTwo.content());
        System.out.println("similarity = " + similarity);
    }

    @Test
    public void should_use_multimodal_embedding_model() {
        var embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        Response<Embedding> embeddingOne = embeddingModel.embed("baby dog");
        Response<Embedding> embeddingTwo = embeddingModel.embed("puppy");
        double similarity = CosineSimilarity.between(embeddingOne.content(), embeddingTwo.content());
        System.out.println("similarity = " + similarity);
    }
}
