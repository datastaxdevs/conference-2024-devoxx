package devoxx.rag;

/**
 * We will interact with the Vertex AI API to get the product recommendation.
 *
 * @param productId
 *      product identifier
 * @param productName
 *      product name
 * @param vector
 *      vector representation of the product
 */
public record Product(String productId, String productName, Object vector) {}
