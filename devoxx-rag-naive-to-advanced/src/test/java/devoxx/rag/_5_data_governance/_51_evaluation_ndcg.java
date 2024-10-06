package devoxx.rag._5_data_governance;

import devoxx.rag.evaluation.RankedResults;
import devoxx.rag.evaluation.ndcg.NormalizedDiscountedCumulativeGain;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _51_evaluation_ndcg {

    @Test
    public void should_explain_ndcg() {
        // Instantiate the evaluator
        NormalizedDiscountedCumulativeGain<String> evaluator = new NormalizedDiscountedCumulativeGain<>();

        // Prepare the list of ranked results
        List<RankedResults<String>> queriesResults = new ArrayList<>();

        // Example data for a single query
        RankedResults<String> rankedResults = new RankedResults<>("What is the capital of France?");

        // Add results (score, document)
        rankedResults.addResult(0.9, "Paris is the capital of France."); // Expected grade: 3
        rankedResults.addResult(0.85, "The capital of Germany is Berlin."); // Expected grade: 0
        rankedResults.addResult(0.7, "Madrid is the capital of Spain."); // Expected grade: 0
        rankedResults.addResult(0.6, "France is a country in Europe."); // Expected grade: 1
        rankedResults.addResult(0.85, "France's capital city is Paris."); // Expected grade: 3

        // Add relevance grades (document, grade)
        rankedResults.addRelevanceGrade("Paris is the capital of France.", 3);
        rankedResults.addRelevanceGrade("France's capital city is Paris.", 3);
        rankedResults.addRelevanceGrade("France is a country in Europe.", 1);

        // Add the ranked results to the list
        queriesResults.add(rankedResults);

        // Define the cutoff rank (e.g., NDCG@3)
        int cutoffRank = 3;

        // Compute NDCG@K
        System.out.println(yellow("\n===== NDCG@15 (static) ===== "));
        System.out.println(cyan("     Used Objects : ") + "String");
        System.out.println(cyan("Relevance Checker : ") + evaluator.getRelevanceChecker().getClass().getSimpleName());
        System.out.println(cyan("      Cutoff Rank : ") + cutoffRank);
        System.out.println(cyan("            Score : ") + evaluator.eval(queriesResults, cutoffRank));

        // Printing the ranked segments
        System.out.println(yellow("\nRanked Segments:"));
        int rank = 1;
        for (Map.Entry<Double, List<String>> entry : rankedResults.getMatches().descendingMap().entrySet()) {
            List<String> docs = entry.getValue();
            for (String doc : docs) {
                if (rank > cutoffRank) {
                    break;
                }
                int grade = rankedResults.getRelevanceGrades().getOrDefault(doc, 0);
                System.out.println("Rank: " + rank + " - Score: " + entry.getKey() + " - Grade: " + grade + " - " + doc);
                rank++;
            }
            if (rank > cutoffRank) {
                break;
            }
        }
    }
}
