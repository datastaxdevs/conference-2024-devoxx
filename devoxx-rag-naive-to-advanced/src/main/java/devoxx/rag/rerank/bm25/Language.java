package devoxx.rag.rerank.bm25;

/**
 * Language enum
 */
public enum Language {

    ENGLISH("en"),
    FRENCH("fr"),
    DANISH("da"),
    DUTCH("nl"),
    FINNISH("fi"),
    GERMAN("de"),
    HUNGARIAN("hu"),
    ITALIAN("it"),
    NORWEGIAN("no"),
    PORTUGUESE("pt"),
    ROMANIAN("ro"),
    RUSSIAN("ru"),
    SPANISH("es"),
    SWEDISH("sv"),
    TURKISH("tr");

    private final String code;

    Language(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}
