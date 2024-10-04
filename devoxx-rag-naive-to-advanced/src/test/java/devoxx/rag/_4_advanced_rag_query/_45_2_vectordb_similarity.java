package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.Collection;
import com.datastax.astra.client.model.Document;
import com.datastax.astra.client.model.FindOptions;
import com.datastax.astra.client.model.Projections;
import com.datastax.astra.client.model.SimilarityMetric;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestClassOrder;
import org.junit.jupiter.api.TestMethodOrder;

import java.time.Instant;
import java.util.Arrays;
import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class _45_2_vectordb_similarity extends AbstractDevoxxTest {

    static Collection<Document> dotProduct;
    static Collection<Document> euclidean;
    static Collection<Document> cosine;

    @Test
    @Order(1)
    public void should_init_stores() {
        System.out.println(yellow("Creating collections..."));
        dotProduct = getAstraDatabase()
                .createCollection("col_dotproduct", 384, SimilarityMetric.DOT_PRODUCT);
        System.out.println(cyan("[OK]") + " col_dotproductDot created");
        euclidean = getAstraDatabase()
                .createCollection("col_euclidean", 384, SimilarityMetric.EUCLIDEAN);
        System.out.println(cyan("[OK]") + " col_euclidean created");
        cosine = getAstraDatabase()
                .createCollection("col_cosine", 384, SimilarityMetric.COSINE);
        System.out.println(cyan("[OK]") + " col_cosine created");
        dotProduct.deleteAll();
        euclidean.deleteAll();
        cosine.deleteAll();
    }

    EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

    List<String> sentences = Arrays.asList(
            "The ancient ruins stood silently beneath the dense canopy of the Amazon rainforest.",
            "Quantum computing could revolutionize data encryption by making current security protocols obsolete.",
            "A bright comet streaked across the night sky, leaving a fiery trail in its wake.",
            "The concept of sustainable urban farming has gained popularity in metropolitan areas.",
            "A bustling street market in Marrakech offers an array of vibrant textiles and exotic spices.",
            "Artificial intelligence is now capable of creating lifelike images from text descriptions.",
            "The architect's innovative design incorporated nature into every aspect of the building.",
            "The global push for renewable energy is reshaping the geopolitics of oil-dependent nations.",
            "A lone musician played a melancholic melody on his violin, echoing through the empty plaza.",
            "The sudden spike in cryptocurrency prices caught many investors off guard."
    );

    @Test
    @Order(2)
    public void should_ingestion() {
        System.out.println(yellow("Ingest sentences..."));
        sentences.stream().map(TextSegment::from).forEach(segment -> {
            Response<Embedding> response = embeddingModel.embed(segment);
            Document doc = Document.create()
                    .vector(response.content().vector())
                    .append("content", segment.text())
                    .append("insertion_date", Instant.now());
            dotProduct.insertOne(doc);
            euclidean.insertOne(doc);
            cosine.insertOne(doc);
        });
    }

    @Test
    @Order(3)
    public void similarity() {
        Collection<Document> dotProduct = getAstraDatabase().getCollection("col_dotproduct");
        String query = "Is Artificial intelligence capable of creating images ?";
        Response<Embedding> questionEmbedding = embeddingModel.embed(TextSegment.from(query));

        System.out.println(yellow("Similarity Result"));
        dotProduct.find(new FindOptions()
                .sort(questionEmbedding.content().vector())
                .projection(Projections.include("*"))
                .includeSimilarity()
                .limit(3)).forEach(doc -> {
            System.out.println(cyan(doc.getSimilarity().get().toString()) + " - " + doc.getString("content"));
        });
    }

    @Test
    @Order(4)
    public void should_conserve_ordering() {

        // Question
        String query = "Is Artificial intelligence capable of creating images ?";
        Response<Embedding> questionEmbedding = embeddingModel.embed(TextSegment.from(query));

        System.out.println(yellow("DOT PRODUCT..."));
        dotProduct.find(questionEmbedding.content().vector(), 3).all().forEach(doc -> {
            System.out.println("- " + doc.getString("content"));
        });

        System.out.println(yellow("EUCLIDEAN..."));
        euclidean.find(questionEmbedding.content().vector(), 3).all().forEach(doc -> {
            System.out.println("- " + doc.getString("content"));
        });

        System.out.println(yellow("COSINE..."));
        cosine.find(questionEmbedding.content().vector(), 3).all().forEach(doc -> {
            System.out.println("- " + doc.getString("content"));
        });
    }

    @Test
    @Order(4)
    public void should_delete_collections() {
        System.out.println(yellow("Deleting collections..."));
        getAstraDatabase().dropCollection("col_dotproduct");
        getAstraDatabase().dropCollection("col_euclidean");
        getAstraDatabase().dropCollection("col_cosine");
    }
}
