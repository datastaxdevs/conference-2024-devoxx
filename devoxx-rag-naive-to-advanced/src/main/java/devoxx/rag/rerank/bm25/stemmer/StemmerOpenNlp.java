package devoxx.rag.rerank.bm25.stemmer;

import devoxx.rag.rerank.bm25.Language;

public class StemmerOpenNlp implements Stemmer{
    @Override
    public String stem(String word) {
        return "";
    }

    @Override
    public Language getSupportedLanguage() {
        return null;
    }
}
