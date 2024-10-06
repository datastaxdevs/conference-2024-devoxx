package devoxx.rag.evaluation;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import lombok.Data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

@Data
public class RankedResults<T> {

    String query;

    TreeMap<Double, List<Embedding>> results;

    List<EmbeddingMatch<T>> matches;
    Set<T> expectedMatches;

    Set<Embedding> groundTruth;

    private Map<Embedding, Integer> relevanceGrades;

    public RankedResults() {
        this.results = new TreeMap<>(Collections.reverseOrder()); // To sort scores in descending order
        this.groundTruth = new HashSet<>();
        this.relevanceGrades = new HashMap<>();
    }

    public RankedResults(String query) {
        this.query = query;
        this.results = new TreeMap<>(Collections.reverseOrder()); // To sort scores in descending order
        this.groundTruth = new HashSet<>();
        this.relevanceGrades = new HashMap<>();
    }

    public void addResults(List<EmbeddingMatch<?>> matches) {
        if (matches == null) {
            return;
        }
        matches.forEach(match -> addResult(match.score(), match.embedding()));
    }

    public void addResult(double score, Embedding doc) {
        results.computeIfAbsent(score, k -> new ArrayList<>()).add(doc);
    }

    public void addGroundTruth(Embedding doc) {
        groundTruth.add(doc);
    }

    public void addRelevanceGrade(Embedding doc, int grade) {
        relevanceGrades.put(doc, grade);
    }
}
