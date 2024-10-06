package devoxx.rag._5_data_governance;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.mrr.MeanReciprocalRank;
import devoxx.rag.evaluation.relevance.EmbeddingSimilarityRelevanceChecker;
import devoxx.rag.similarity.CosineSimilarity;
import devoxx.rag.similarity.EuclideanSimilarity;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.TreeMap;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _50_evaluation_mrr extends AbstractDevoxxTest {

    @Test
    public void should_explain_mrr() {
          // We are working with STRING
          MeanReciprocalRank<String> mmr15 = new MeanReciprocalRank<>();
          // Question
          RankedResults<String> rankedResults = new RankedResults<>("What is the capital of France?");
          // Target responses (Also expressed as Strings)
          rankedResults.addExpectedAnswers("Paris is the capital of France.");
          rankedResults.addExpectedAnswers("France's capital city is Paris.");
          // Results from query
          rankedResults.addResult(0.9, "Paris is the capital of France.");
          rankedResults.addResult(0.85, "The capital of Germany is Berlin.");
          rankedResults.addResult(0.7, "Madrid is the capital of Spain.");
          rankedResults.addResult(0.6, "France is a country in Europe.");
          // The default RELEVANCE FUNCTION IS EXACT MATCHING.
          // As one of the results is an exact match, the MRR will be 1.0 for this rank.
          double mrr = new MeanReciprocalRank<String>().eval(List.of(rankedResults));
          System.out.println(yellow("===== MRR@15 ===== "));
          System.out.println(cyan("     Used Objects : ") + "String");
          System.out.println(cyan("Relevance Checker : ") + mmr15.getRelevanceChecker().getClass().getSimpleName());
          System.out.println(cyan("           Score  : ") + mrr);
    }

    @Test
    public void should_explain_mrr_embeddings() {
        // We are working with STRING
        EmbeddingSimilarityRelevanceChecker relevance = new EmbeddingSimilarityRelevanceChecker(
                new CosineSimilarity(), 0.89);
        MeanReciprocalRank<Embedding> mmr15 = new MeanReciprocalRank<>(relevance, 15);

        // Question
        RankedResults<Embedding> rankedResults = new RankedResults<>("What is the capital of France?");
        // Target responses (Also expressed as Embeddings now)
        rankedResults.addExpectedAnswers(embed("Paris is the capital of France."));
        rankedResults.addExpectedAnswers(embed("France's capital city is Paris."));
        // Results from query
        rankedResults.addResult(0.9, embed("Paris is the capital of France."));
        rankedResults.addResult(0.85, embed("The capital of Germany is Berlin."));
        rankedResults.addResult(0.7, embed("Madrid is the capital of Spain."));
        rankedResults.addResult(0.6, embed("France is a country in Europe."));
        // The default RELEVANCE FUNCTION IS EXACT MATCHING.
        // As one of the results is an exact match, the MRR will be 1.0 for this rank.
        double mrr = mmr15.eval(rankedResults);
        System.out.println(yellow("===== MRR@15 ===== "));
        System.out.println(cyan("     Used Objects : ") + "Embedding");
        System.out.println(cyan("Relevance Checker : ") + mmr15.getRelevanceChecker().getClass().getSimpleName());
        System.out.println(cyan("           Score  : ") + mrr);
    }

    Embedding embed(String text) {
        return getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed(text).content();
    }


    @Test
    public void should_mmr_for_rag() {

        String question = "What is the population of Berlin?";

        // We are working with STRING
        EmbeddingSimilarityRelevanceChecker relevance = new EmbeddingSimilarityRelevanceChecker(
                new CosineSimilarity(), 0.75);
        MeanReciprocalRank<Embedding> mmr15 = new MeanReciprocalRank<>(relevance, 15);

        // Question
        RankedResults<Embedding> rankedResults = new RankedResults<>(question);
        // What I am supposed the get
        rankedResults.addExpectedAnswers(embed("Berlin is about 3.85 millions inhabitants"));
        rankedResults.addExpectedAnswers(embed("The population of Berlin is 3.85 millions"));

        // RAG
        var embeddingStore = new AstraDbEmbeddingStore(createCollection("berlin", 768));
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddingStore.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(embed(question))
                        .minScore(0.1)
                        .maxResults(15)
                        .build()).matches();

        relevantEmbeddings.forEach(embeddingMatch ->
                rankedResults.addResult(embeddingMatch.score(), embeddingMatch.embedding()));

        // Compute MRR
        double mrr = mmr15.eval(rankedResults);
        System.out.println(yellow("===== MRR@15 (rag) ===== "));
        System.out.println(cyan("     Used Objects : ") + "Embedding");
        System.out.println(cyan("Relevance Checker : ") + mmr15.getRelevanceChecker().getClass().getSimpleName());
        System.out.println(cyan("           Score  : ") + mrr);

        // Remove previous results
        rankedResults.getMatches().clear();

        // Re-Ranking
        TreeMap<Double, TextSegment> rankedSegments = new TreeMap<>();
        relevantEmbeddings.stream()
                .forEach(match -> {
                    Double score = getScoringModel().score(match.embedded().text(), question).content();
                    rankedResults.addResult(score, match.embedding());
                });
        System.out.println(yellow("\n===== MRR@15 (rag + re-rank) ===== "));
        System.out.println(cyan("     Used Objects : ") + "Embedding");
        System.out.println(cyan("Relevance Checker : ") + mmr15.getRelevanceChecker().getClass().getSimpleName());
        System.out.println(cyan("      Score (new) : ") + mmr15.eval(rankedResults));
    }
}
