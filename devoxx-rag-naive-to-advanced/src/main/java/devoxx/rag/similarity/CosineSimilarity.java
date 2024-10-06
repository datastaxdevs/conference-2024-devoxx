package devoxx.rag.similarity;

/**
 * Implementation of the Cosine Similarity
 */
public class CosineSimilarity implements Similarity {


    public double compute(float[] vectorA, float[] vectorB) {
        validate(vectorA, vectorB);
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; ++i) {
            dotProduct += (double) (vectorA[i] * vectorB[i]);
            normA += (double) (vectorA[i] * vectorA[i]);
            normB += (double) (vectorB[i] * vectorB[i]);
        }
        return dotProduct / Math.max(Math.sqrt(normA) * Math.sqrt(normB), 9.99999993922529E-9);
    }

}
