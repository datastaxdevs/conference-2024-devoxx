package devoxx.rag.rerank.bm25.stemmer;

import devoxx.rag.rerank.bm25.Language;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.danishStemmer;
import org.tartarus.snowball.ext.dutchStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.finnishStemmer;
import org.tartarus.snowball.ext.frenchStemmer;
import org.tartarus.snowball.ext.germanStemmer;
import org.tartarus.snowball.ext.hungarianStemmer;
import org.tartarus.snowball.ext.italianStemmer;
import org.tartarus.snowball.ext.norwegianStemmer;
import org.tartarus.snowball.ext.porterStemmer;
import org.tartarus.snowball.ext.portugueseStemmer;
import org.tartarus.snowball.ext.romanianStemmer;
import org.tartarus.snowball.ext.russianStemmer;
import org.tartarus.snowball.ext.spanishStemmer;
import org.tartarus.snowball.ext.swedishStemmer;
import org.tartarus.snowball.ext.turkishStemmer;

public class StemmerSnowBall implements Stemmer {

    private final SnowballStemmer stemmer;

    private final Language language;

    public StemmerSnowBall(Language language) {
        this.language = language;
        switch (language) {
            case ENGLISH -> stemmer = new englishStemmer();
            case FRENCH -> stemmer = new frenchStemmer();
            case DANISH -> stemmer = new danishStemmer();
            case DUTCH -> stemmer = new dutchStemmer();
            case FINNISH -> stemmer = new finnishStemmer();
            case GERMAN -> stemmer = new germanStemmer();
            case HUNGARIAN -> stemmer = new hungarianStemmer();
            case ITALIAN -> stemmer = new italianStemmer();
            case NORWEGIAN -> stemmer = new norwegianStemmer();
            case PORTUGUESE -> stemmer = new portugueseStemmer();
            case ROMANIAN -> stemmer = new romanianStemmer();
            case RUSSIAN -> stemmer = new russianStemmer();
            case SPANISH -> stemmer = new spanishStemmer();
            case SWEDISH -> stemmer = new swedishStemmer();
            case TURKISH -> stemmer = new turkishStemmer();
            default -> stemmer = new porterStemmer();
        }
    }

    /** {@inheritDoc} */
    @Override
    public String stem(String word) {
        stemmer.setCurrent(word);
        if (stemmer.stem()) {
            return stemmer.getCurrent();
        } else {
            return word;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Language getSupportedLanguage() {
        return Language.ENGLISH;
    }

    /**
     * Gets stemmer
     *
     * @return value of stemmer
     */
    public SnowballStemmer getStemmer() {
        return stemmer;
    }

    /**
     * Gets language
     *
     * @return value of language
     */
    public Language getLanguage() {
        return language;
    }
}
