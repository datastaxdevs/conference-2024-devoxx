package devoxx.rag._4_advanced_rag_query;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Bm25ReRanking {

    private List<String> documents;
    private Map<String, Integer> docFreqs;
    private List<Map<String, Integer>> termFreqs;
    private double avgDocLength;
    private int totalDocs;
    private double k1 = 1.5;
    private double b = 0.75;

    public Bm25ReRanking(List<String> documents) {
        this.documents = documents;
        this.totalDocs = documents.size();
        this.termFreqs = new ArrayList<>();
        this.docFreqs = new HashMap<>();
        int totalLength = 0;

        // Preprocess documents
        for (String doc : documents) {
            String[] tokens = doc.split("\\s+");
            totalLength += tokens.length;
            Map<String, Integer> freqs = new HashMap<>();
            Set<String> uniqueTerms = new HashSet<>();
            for (String term : tokens) {
                term = term.toLowerCase(); // Optional: normalize case
                freqs.put(term, freqs.getOrDefault(term, 0) + 1);
                uniqueTerms.add(term);
            }
            termFreqs.add(freqs);

            for (String term : uniqueTerms) {
                docFreqs.put(term, docFreqs.getOrDefault(term, 0) + 1);
            }
        }
        this.avgDocLength = (double) totalLength / totalDocs;
    }

    public double score(String query, int docIndex) {
        Map<String, Integer> freqs = termFreqs.get(docIndex);
        int docLength = freqs.values().stream().mapToInt(Integer::intValue).sum();
        double score = 0.0;

        String[] queryTerms = query.split("\\s+");
        for (String term : queryTerms) {
            term = term.toLowerCase(); // Optional: normalize case
            if (!freqs.containsKey(term)) continue;
            int f = freqs.get(term);
            int df = docFreqs.getOrDefault(term, 0);
            double idf = Math.log(1 + (totalDocs - df + 0.5) / (df + 0.5));
            double numerator = f * (k1 + 1);
            double denominator = f + k1 * (1 - b + b * docLength / avgDocLength);
            score += idf * (numerator / denominator);
        }
        return score;
    }

    public List<Integer> rank(String query) {
        List<Integer> docIndices = new ArrayList<>();
        List<Double> scores = new ArrayList<>();

        for (int i = 0; i < documents.size(); i++) {
            double score = score(query, i);
            docIndices.add(i);
            scores.add(score);
        }

        // Sort documents by score in descending order
        docIndices.sort((i1, i2) -> Double.compare(scores.get(i2), scores.get(i1)));

        return docIndices;
    }
}




