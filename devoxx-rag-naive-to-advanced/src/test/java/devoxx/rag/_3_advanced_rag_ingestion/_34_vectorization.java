package devoxx.rag._3_advanced_rag_ingestion;

import ai.djl.sentencepiece.SpTokenizer;
import org.junit.jupiter.api.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

public class _34_vectorization {

    String chunk = "Devoxx is the greatest conference to developer to know about the latest innovations in the tech world";

    @Test
    public void bagOfWordTokenization() {
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
    public void gemini_tokenizer() throws IOException {
        Path modelFile = Paths.get("src/test/resources/gemini/gemini-tokenizer.model");
        if (Files.notExists(modelFile)) {
            throw new FileNotFoundException("Model file not found");
        }

        String textToTokenize = """
            When integrating an LLM into your application to extend it and make it smarter, \
            it's important to be aware of the pitfalls and best practices you need to follow \
            to avoid some common problems and integrate them successfully. This article will \
            guide you through some key best practices that I've come across.
            """;

        byte[] modelFileBytes = Files.readAllBytes(modelFile);
        try (SpTokenizer tokenizer = new SpTokenizer(modelFileBytes)) {
            List<String> tokens = tokenizer.tokenize(textToTokenize);
            for (String token: tokens) {
                System.out.format("[%s]%n", token);
            }

            System.out.println("Token count: " + tokens.size());
            assertThat(tokens.size()).isEqualTo(61);
        }
    }


}
