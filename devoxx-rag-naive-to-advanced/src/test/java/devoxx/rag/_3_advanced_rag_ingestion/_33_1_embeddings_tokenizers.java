package devoxx.rag._3_advanced_rag_ingestion;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.sentencepiece.SpTokenizer;
import org.junit.jupiter.api.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class _33_1_embeddings_tokenizers {

    String chunk = "Devoxx is the greatest conference for developers to know about " +
            "the latest innovations in the tech world";

    @Test
    public void bertTokenization() throws Exception {
        List<String> vocabList = Files.readAllLines(Paths.get("src/test/resources/text/vocab.txt"));
        // Create a vocabulary using DefaultVocabulary
        Vocabulary vocab = new DefaultVocabulary(vocabList);
        // Initialize BERT tokenizer
        BertFullTokenizer tokenizer = new BertFullTokenizer(vocab, true);

        String textToTokenize = """
            When integrating an LLM into your application to extend it and make it smarter, \
            it's important to be aware of the pitfalls and best practices you need to follow \
            to avoid some common problems and integrate them successfully. This article will \
            guide you through some key best practices that I've come across.
            """;

        String question = "What this paragraph is about ?";

        // Tokenize the input text
        List<String> pTokens = tokenizer.tokenize(textToTokenize);
        List<String> qTokens = tokenizer.tokenize(question);

        // Add special tokens according to the format: [CLS] Question [SEP] Paragraph [SEP]
        List<String> tokens = new ArrayList<>();
        tokens.add("[CLS]");
        tokens.addAll(qTokens);
        tokens.add("[SEP]");
        tokens.addAll(pTokens);
        tokens.add("[SEP]");

        // Map tokens to their indices in the vocabulary using a custom method
        List<Long> inputIds = convertTokensToIds(vocab, tokens);

        // Create token type IDs (Segment IDs): 0 for question, 1 for paragraph
        List<Long> tokenTypeIds = new ArrayList<>();
        for (int i = 0; i < qTokens.size() + 2; i++) { // [CLS] + Question Tokens + [SEP]
            tokenTypeIds.add(0L);
        }
        for (int i = 0; i < pTokens.size() + 1; i++) { // Paragraph Tokens + [SEP]
            tokenTypeIds.add(1L);
        }

        System.out.println("Tokens: " + pTokens);
        System.out.println("Tokens Index IDs: " + inputIds);

        // Bert Inference per token

        // Pooling


    }

    /**
     * Converts a list of tokens into a list of token IDs using the provided vocabulary.
     *
     * @param vocab  the vocabulary to use for mapping tokens to IDs
     * @param tokens the list of tokens to be converted
     * @return a list of IDs corresponding to each token
     */
    public static List<Long> convertTokensToIds(Vocabulary vocab, List<String> tokens) {
        List<Long> ids = new ArrayList<>();
        for (String token : tokens) {
            long id = vocab.getIndex(token); // Get the ID of the token from the vocabulary
            if (id == -1) { // If the token is not found, replace it with [UNK] token ID
                id = vocab.getIndex("[UNK]");
            }
            ids.add(id);
        }
        return ids;
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
