package devoxx.rag.similarity;

/**
 * Implementation of DotProduct Similarity
 */
public class DotProductSimilarity implements Similarity {

    /** {@inheritDoc} */
    public double compute(float[] vectorA, float[] vectorB) {
        validate(vectorA, vectorB);
        double sum = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            sum += Math.pow(vectorA[i] - vectorB[i], 2);
        }
        return Math.sqrt(sum);
    }

}
