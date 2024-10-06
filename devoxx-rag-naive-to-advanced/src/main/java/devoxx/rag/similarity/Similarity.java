package devoxx.rag.similarity;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.internal.Exceptions;
import dev.langchain4j.internal.ValidationUtils;

public interface Similarity {

    double compute(float[] vectorA, float[] vectorB);

    default double compute(Embedding embeddingA,Embedding embeddingB) {
        ValidationUtils.ensureNotNull(embeddingA, "embeddingA");
        ValidationUtils.ensureNotNull(embeddingA, "embeddingA");
        return compute(embeddingA.vector(), embeddingB.vector());
    }

    default void validate(float[] vectorA, float[] vectorB) {
        ValidationUtils.ensureNotNull(vectorA, "vectorA");
        ValidationUtils.ensureNotNull(vectorB, "vectorB");
        if (vectorA.length != vectorB.length) {
            throw Exceptions.illegalArgument("Length of vector a (%s) must be equal to the " +
                    "length of vector b (%s)", new Object[]{vectorA.length, vectorB.length});
        }
    }

}
