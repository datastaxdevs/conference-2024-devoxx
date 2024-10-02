package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.store.embedding.CosineSimilarity;
import devoxx.rag.AbstracDevoxxSampleTest;
import devoxx.rag.experiments.Utils;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.datastax.astra.internal.utils.AnsiUtils.magenta;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _39_semantic_chunking extends AbstracDevoxxSampleTest {
    @Test
    public void semanticChunking() {
        String text = loadDocumentText("text/berlin.txt").text();

        var embeddingModel = getEmbeddingModel("text-embedding-004");

        // split by sentences
        DocumentBySentenceSplitter splitter = new DocumentBySentenceSplitter(200, 20);
        List<String> sentences = Arrays.asList(splitter.split(text));

        // create groups of sentences (1 before, 2 after current sentence)
        List<List<String>> slidingWindowSentences = Utils.slidingWindow(sentences, 1, 2);
        List<TextSegment> concatenatedSentences = slidingWindowSentences.stream()
            .map(strings -> TextSegment.from(String.join(" ", strings)))
            .toList();

        // calculate vector embeddings for each of these sentence groups
        List<Embedding> embeddings = embeddingModel.embedAll(concatenatedSentences).content();

        // calculate the pair-wise similarities between each sentence groups
        List<Double> similarities = new ArrayList<>();
        for (int i = 0; i < embeddings.size() - 1; i++) {
            similarities.add(CosineSimilarity.between(embeddings.get(i), embeddings.get(i + 1)));
        }

        // find the 100 lowest similarities
        List<Double> lowestSimilarities = similarities.stream()
            .sorted()
            .limit(100)
            .toList();

        // find the breakpoints (indices) where the similarity is the lowest
        int[] lowestSimilaritiesIndices = lowestSimilarities.stream()
            .mapToInt(similarity ->
                IntStream.range(0, similarities.size())
                    .filter(streamIndex -> similarities.get(streamIndex).equals(similarity))
                    .findFirst()
                    .orElse(-1))
            .sorted()
            .toArray();

        System.out.println(magenta("Lowest similarity breakpoints = ") + Arrays.toString(lowestSimilaritiesIndices));

        List<String> finalSentenceGroups = new ArrayList<>();

        int startIndex = 0;
        for (int lowestSimilaritiesIndex : lowestSimilaritiesIndices) {
            finalSentenceGroups.add(sentences.subList(startIndex, lowestSimilaritiesIndex)
                .stream()
                .collect(Collectors.joining(" ")));
            startIndex = lowestSimilaritiesIndex;
        }
        finalSentenceGroups.add(sentences.subList(startIndex, sentences.size())
            .stream()
            .collect(Collectors.joining(" ")));

        ScoringModel scoringModel = getScoringModel();

        finalSentenceGroups.forEach(sentenceGroup -> {
            Double score = scoringModel.score(sentenceGroup, "What is the population of Berlin?").content();
            if (score > 0.7) {
                System.out.format(yellow("—".repeat(10) + " Ranking score: %s " + "—".repeat(60)) + "%n %s %n", score, sentenceGroup);
            }
        });
    }
}
