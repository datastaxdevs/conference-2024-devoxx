package devoxx.rag._4_advanced_rag_query;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RRFReRanking {

    public static Map<String, Double> reciprocalRankFusion(List<String> list1, List<String> list2, double k) {
        // Initialize a map to store the cumulative RRF scores
        Map<String, Double> rrfScores = new HashMap<>();

        // Process the first list and assign RRF scores
        for (int rank = 0; rank < list1.size(); rank++) {
            String doc = list1.get(rank);
            rrfScores.put(doc, rrfScores.getOrDefault(doc, 0.0) + 1.0 / (k + rank + 1.0));
        }

        // Process the second list and update RRF scores
        for (int rank = 0; rank < list2.size(); rank++) {
            String doc = list2.get(rank);
            rrfScores.put(doc, rrfScores.getOrDefault(doc, 0.0) + 1.0 / (k + rank + 1.0));
        }

        return rrfScores;
    }
}
