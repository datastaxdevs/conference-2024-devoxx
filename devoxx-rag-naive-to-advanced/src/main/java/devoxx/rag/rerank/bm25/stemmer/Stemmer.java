package devoxx.rag.rerank.bm25.stemmer;

import devoxx.rag.rerank.bm25.Language;

/**
 * A language stemmer.
 */
public interface Stemmer {

    /**
     * Stem a word.
     *
     * @param word
     *      word
     * @return
     *      stemmed word
     */
    String stem(String word);

    /**
     * Get Stemmer from Language
     * @return
     */
    Language getSupportedLanguage();
}
