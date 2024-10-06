package devoxx.rag.rerank.bm25.stopwords;

import com.fasterxml.jackson.databind.ObjectMapper;
import devoxx.rag.rerank.bm25.Language;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Extract the Stop words from the iso file.
 */
public class StopWords {

    public static final Map<String, List<String>> stopWords;

    /**
     * Hide default constructor.
     */
    private StopWords() {
    }

    static {
        try {
            String filename = "stopwords-iso.json";
            ObjectMapper om = new ObjectMapper();
            stopWords = om.readValue(StopWords.class.getClassLoader().getResourceAsStream(filename), Map.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Set<String> getStopWords(Language language) {
        if (stopWords.containsKey(language.getCode())) {
            return Set.copyOf(stopWords.get(language.getCode()));
        } else {
            return Set.of();
        }
    }

}
