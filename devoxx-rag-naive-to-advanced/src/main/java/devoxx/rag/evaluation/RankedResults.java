package devoxx.rag.evaluation;

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

    private String query;

    private TreeMap<Double, List<T>> matches;

    private Set<T> expectedAnswers;

    private Map<T, Integer> relevanceGrades;

    public RankedResults() {
        this.matches = new TreeMap<>(Collections.reverseOrder()); // To sort scores in descending order
        this.expectedAnswers = new HashSet<>();
        this.relevanceGrades = new HashMap<>();
    }

    public RankedResults(String query) {
        this.query = query;
        this.matches = new TreeMap<>(Collections.reverseOrder()); // To sort scores in descending order
        this.expectedAnswers = new HashSet<>();
        this.relevanceGrades = new HashMap<>();
    }

    // tentative
    public void addResults(List<EmbeddingMatch<T>> matches) {
        if (matches == null) {
            return;
        }
        matches.forEach(match -> addResult(match.score(), match.embedded()));
    }

    public void addResult(double score, T doc) {
        matches.computeIfAbsent(score, k -> new ArrayList<>()).add(doc);
    }

    public void addExpectedAnswers(T doc) {
        expectedAnswers.add(doc);
    }

    public void addRelevanceGrade(T doc, int grade) {
        relevanceGrades.put(doc, grade);
    }
}
