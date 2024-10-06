package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.model.Filters;
import com.datastax.astra.internal.utils.AnsiUtils;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.rerank.rrf.ReciprocalRankFusion;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class _46_reranking_rrf extends AbstractDevoxxTest  {

    private static final String COLLECTION_NAME = "quote";

    @Test
    public void testRerankingRRF() {
        String question = "We struggle all our life for nothing";
        Embedding questionEmbedding = getEmbeddingModel(MODEL_EMBEDDING_TEXT).embed(question).content();
        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(getCollection(COLLECTION_NAME));

        List<EmbeddingMatch<TextSegment>> aristotleResults = embeddingStore.findRelevant(
                questionEmbedding,
                Filters.eq("authors", "aristotle"),
                10, 0.1d);

        List<EmbeddingMatch<TextSegment>> plateResults     = embeddingStore.findRelevant(
                questionEmbedding,
                Filters.eq("authors", "plato"),
                10, 0.1d);

        System.out.println(AnsiUtils.yellow("=========== ARISTOTLE ============"));
        aristotleResults.forEach(match -> {
            System.out.println(AnsiUtils.cyan(BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP).toString()) + " - " + match.embedded().text());
        });
        System.out.println(AnsiUtils.yellow("============= PLATO =============="));
        plateResults.forEach(match -> {
            System.out.println(AnsiUtils.cyan(BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP).toString()) + " - " + match.embedded().text());
        });

        // RRF
        List<TextSegment> aristotleList = aristotleResults.stream().map(EmbeddingMatch::embedded).toList();
        List<TextSegment> plateList     = plateResults.stream().map(EmbeddingMatch::embedded).toList();
        Map<TextSegment, Double> fusedResults = new ReciprocalRankFusion().score(Arrays.asList(aristotleList, plateList));
        System.out.println(AnsiUtils.yellow("============= RRF =============="));
        fusedResults.entrySet()
                .stream()
                .sorted((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()))
                .forEach(entry -> {
                    System.out.println(AnsiUtils.cyan(BigDecimal.valueOf(entry.getValue()).setScale(4, RoundingMode.HALF_UP).toString()) + " - " + entry.getKey().text());
                });

    }
}
