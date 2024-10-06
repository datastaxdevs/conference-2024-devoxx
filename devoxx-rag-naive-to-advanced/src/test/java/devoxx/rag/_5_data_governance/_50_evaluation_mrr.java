package devoxx.rag._5_data_governance;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.model.vertexai.VertexAiScoringModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.mrr.MeanReciprocalRank;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.TreeMap;

import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _50_evaluation_mrr extends AbstractDevoxxTest {

    static final MeanReciprocalRank MRR = new MeanReciprocalRank();

    @Test
    public void should_explain_mrr() {
        
//        RankedResults<String> rankedResults = new RankedResults<>();
//        // Question
//        rankedResults.setQuery("What is the capital of France?");
//
//        // Target responses
//        rankedResults.addGroundTruth("Paris is the capital of France.");
//        rankedResults.addGroundTruth("France's capital city is Paris.");
//
//        // Results from query
//        getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed("Paris is the capital of France.").content();
//        rankedResults.addResult(0.9, "Paris is the capital of France.");
//        rankedResults.addResult(0.85, "The capital of Germany is Berlin.");
//        rankedResults.addResult(0.7, "Madrid is the capital of Spain.");
//        rankedResults.addResult(0.6, "France is a country in Europe.");
//
//        // Compute MRR
//        double mrr = MRR.eval(List.of(rankedResults));
//        System.out.println("MRR@15: " + mrr);
    }
    /*
    @Test
    public void should_mmr() {

        RankedResults rankedResults = new RankedResults<>();
        // Question
        getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed("What is the population of Berlin?").content();
        rankedResults.setQuery("What is the population of Berlin?");
        rankedResults.addGroundTruth("Berlin is about 3.85 millions inhabitants");
        rankedResults.addGroundTruth("The population of Berlin is 3.85 millions");

        rankedResults.addResult(0.90674865, "Berlin is the capital and largest city of Germany, " +
                "both by area and by population. Its more than 3.85 million inhabitants make it the European " +
                "Union's most populous city, as measured by population within city limits.");
        rankedResults.addResult(0.88403654, "At the end of 2018, the city-state of Berlin had 3.75 million " +
                "registered inhabitants in an area of 891.1 km2 (344.1 sq mi). The city's population density was 4,206 " +
                "inhabitants per km2. Berlin is the most populous city proper in the European Union.");
        rankedResults.addResult(0.88242877, "Foreign residents of Berlin originate from about 190 countries. " +
                "48 percent of the residents under the age of 15 have a migration background in 2017. Berlin in 2009 was " +
                "estimated to have 100,000 to 250,000 unregistered inhabitants.");
        rankedResults.addResult(0.8757057, "In 2019, the urban area of Berlin had about 4.5 million inhabitants." +
                " As of 2019, the functional urban area was home to about 5.2 million people. The entire Berlin-Brandenburg " +
                "capital region has a population of more than 6 million in an area of 30,546 km2 (11,794 sq mi).");

        // Compute MRR
        double mrr = MRR.eval(List.of(rankedResults));
        System.out.println("MRR@15: " + mrr);
    }


    @Test
    public void should_demo_mrr() {
        // Ingest large amount of data
        var embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);
        var embeddingStore = new AstraDbEmbeddingStore(createCollection("berlin", 768));

        //getCollection("berlin").deleteAll();
        //ingestDocument("text/berlin.txt", embeddingModel, embeddingStore);
        //System.out.println(cyan("[OK] ") + "Ingested Berlin document");

        System.out.println(yellow("Question: "));
        String question = "What is the population of Berlin?";

        System.out.println(yellow("Similarity: "));
        TreeMap<Double, TextSegment> similaritySegments = new TreeMap<>();
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(embeddingModel.embed(question).content())
                        .minScore(0.1)
                        .maxResults(15)
                        .build())
                        .matches();
        relevantEmbeddings.forEach(embeddingMatch -> {
            similaritySegments.put(embeddingMatch.score(), embeddingMatch.embedded());
        });

        RankedResults<String> rankedResults = new RankedResults<>();
        rankedResults.setQuery(question);
        rankedResults.addGroundTruth("Berlin is about 3.85 millions inhabitants");
        rankedResults.addGroundTruth("The population of Berlin is 3.85 millions");
        similaritySegments.descendingMap().forEach((similarity, textSegment) -> {
            rankedResults.addResult(similarity, textSegment.text());
            System.out.println("Score: " + similarity + " - " + textSegment.text());
        });

        // Compute MRR
        double mrr = new MeanReciprocalRank<String>().eval(List.of(rankedResults), 15);
        System.out.println("MRR@15[similarity-search]: " + mrr);

        System.out.println(yellow("Reranking: "));
        ScoringModel reranker = VertexAiScoringModel.builder()
                .projectId(System.getenv("GCP_PROJECT_ID"))
                .projectNumber(System.getenv("GCP_PROJECT_CODE"))
                .location(System.getenv("GCP_LOCATION"))
                .model("semantic-ranker-512")
                .build();

        TreeMap<Double, TextSegment> rankedSegments = new TreeMap<>();
        relevantEmbeddings.stream()
                .map(EmbeddingMatch::embedded)
                .forEach(textSegment -> {
                    Double score = reranker.score(textSegment.text(), question).content();
                    rankedSegments.put(score, textSegment);
                });
        RankedResults<String> result2 = new RankedResults<>();
        result2.setQuery(question);
        result2.addGroundTruth("The population of Berlin is about 3.85 million inhabitants");
        result2.addGroundTruth("The population of Berlin is 3.85 millions");
        rankedSegments.descendingMap().forEach((similarity, textSegment) -> {
            result2.addResult(similarity, textSegment.text());
        });
        // Compute MRR
        double mrr2 = new MeanReciprocalRank<String>().eval(List.of(result2), 15);
        System.out.println("MRR@15[similarity-search + re-rerank]: " + mrr2);
    }
*/
}
