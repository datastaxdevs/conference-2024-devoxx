package devoxx.rag._2_ingestion._4_embedding;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

public class _02_Tokenization {

    String chunk = "Devoxx is the greatest conference to developer to know about the latest innovations in the tech world";

    @Test
    public void bagOfWorldTokenization() {
        // Convert the sentence to lowercase and split by space to get words
        String[] words = chunk.toLowerCase().split(" ");
        // Create a Bag of Words using a HashMap
        Map<String, Integer> bagOfWords = new HashMap<>();
        for (String word : words) {
            // Count frequency of each word
            bagOfWords.put(word, bagOfWords.getOrDefault(word, 0) + 1);
        }
        // Display the results
        System.out.println("Bag of Words representation:");
        for (Map.Entry<String, Integer> entry : bagOfWords.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    @Test
    public void word2VectTokenization() {
//        File file = new File("corpus.txt");
//        java.nio.file.Files.write(file.toPath(), Arrays.asList(sentence));
//
//        // Set up a sentence iterator to read the corpus
//        BasicLineIterator iterator = new BasicLineIterator(file);
//
//        // Use a tokenizer factory with a common preprocessor
//        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
//        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
//
//        // Build and train the Word2Vec model
//        Word2Vec word2Vec = new Word2Vec.Builder()
//                .minWordFrequency(1)
//                .iterations(5)
//                .layerSize(100)
//                .seed(42)
//                .windowSize(5)
//                .iterate(iterator)
//                .tokenizerFactory(tokenizerFactory)
//                .build();
//
//        System.out.println("Training Word2Vec model...");
//        word2Vec.fit();
//
//        // Get the vector for a specific word
//        double[] vector = word2Vec.getWordVector("developer");
//        System.out.println("Word2Vec representation for 'developer': " + Arrays.toString(vector));
    }


    @Test
    public void bertTokenization() {
    }
    @Test
    public void mapTokenToIds() {

    }

    @Test
    public void word_embeddings() {
        // Tokenize the chunk
        String[] words = chunk.split(" ");
        for (String word : words) {
            System.out.println(word);
        }
    }

    @Test
    public void gemeni_tokenizer() {
        // Tokenize the chunk
        String[] words = chunk.split(" ");
        for (String word : words) {
            System.out.println(word);
        }
    }


}
