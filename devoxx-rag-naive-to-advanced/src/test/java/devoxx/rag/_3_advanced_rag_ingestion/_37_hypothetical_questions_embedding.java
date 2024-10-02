package devoxx.rag._3_advanced_rag_ingestion;

import com.datastax.astra.client.DataAPIClient;
import com.datastax.astra.client.Database;
import com.datastax.astra.client.model.SimilarityMetric;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import com.google.cloud.vertexai.api.Schema;
import com.google.cloud.vertexai.api.Type;
import com.google.gson.Gson;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.datastax.astra.internal.utils.AnsiUtils.*;

public class _37_hypothetical_questions_embedding extends AbstracDevoxxSampleTest {

    private static final Gson gson = new Gson();
    private static final String PARAGRAPH_KEY = "paragraph";
    private static final String COLLECTION_NAME = "berlin_hypothetical_questions";
    private static final Database DATABASE = new DataAPIClient(ASTRA_TOKEN).getDatabase(ASTRA_API_ENDPOINT);

    private boolean collectionExists() {
        return DATABASE.collectionExists(COLLECTION_NAME);
    }

    private EmbeddingStore<TextSegment> getEmbeddingStore() {
        if (collectionExists()) {
            System.out.println(cyan("Collection already exists."));
            return new AstraDbEmbeddingStore(DATABASE.getCollection(COLLECTION_NAME));
        } else {
            System.out.println(cyan("Creating collection..."));
            return new AstraDbEmbeddingStore(DATABASE.createCollection(COLLECTION_NAME, 768, SimilarityMetric.COSINE));
        }
    }

    @Order(2)
//    @Test
    public void deleteCollection() {
        if (collectionExists()) {
            DATABASE.dropCollection(COLLECTION_NAME);
        }
    }

    @Order(1)
    @Test
    public void ingestionOfHypotheticalQuestions() {
        if (!collectionExists()) {
            System.out.println(cyan("Ingesting hypothetical questions..."));

            Document documentAboutBerlin = loadDocumentText("text/berlin.txt");

            VertexAiGeminiChatModel gemini = VertexAiGeminiChatModel.builder()
                .project(System.getenv("GCP_PROJECT_ID"))
                .location(System.getenv("GCP_LOCATION"))
                .modelName("gemini-1.5-flash-002")
                .maxRetries(5)
                .responseSchema(Schema.newBuilder()
                    .setType(Type.ARRAY)
                    .setItems(Schema.newBuilder().setType(Type.STRING).build())
                    .build())
                .build();

            // ========================================================
            // Prepare 10 questions for each paragraph thanks to an LLM

            List<QuestionParagraph> allQuestionParagraphs = new ArrayList<>();

            DocumentByParagraphSplitter splitter = new DocumentByParagraphSplitter(2000, 100);
            List<TextSegment> paragraphs = splitter.split(documentAboutBerlin);

            for (TextSegment paragraphSegment : paragraphs) {
                System.out.println(cyan("\n==== PARAGRAPH ==================================\n") + paragraphSegment.text());

                try {Thread.sleep(2000);} catch (InterruptedException e) {}

                Response<AiMessage> aiResult = gemini.generate(List.of(
                    SystemMessage.from("""
                        Suggest 10 clear questions whose answer could be given by the user provided text.
                        Don't use pronouns, be explicit about the subjects and objects of the question.
                        """),
                    UserMessage.from(paragraphSegment.text())
                ));
                String[] questions = gson.fromJson(aiResult.content().text(), String[].class);

                System.out.println(yellow("\nQUESTIONS:\n"));
                for (int i = 0; i < questions.length; i++) {
                    String question = questions[i];
                    System.out.println((i + 1) + ") " + question);

                    allQuestionParagraphs.add(new QuestionParagraph(question, paragraphSegment));
                }
            }

            // ===============================================
            // Embed all the pairs of questions and paragraphs

            List<TextSegment> embeddedSegments = allQuestionParagraphs.stream()
                .map(questionParagraph -> TextSegment.from(
                    questionParagraph.question(),
                    new Metadata().put(PARAGRAPH_KEY, questionParagraph.paragraph().text())))
                .toList();

            var embeddingModel = getEmbeddingModel("text-embedding-004");
            var embeddingStore = getEmbeddingStore();

            List<Embedding> embeddings = embeddingModel.embedAll(embeddedSegments).content();
            embeddingStore.addAll(embeddings, embeddedSegments);
        }
    }

    @Order(2)
    @Test
    public void hypotheticalQuestions() {
        // =========================================================
        // Search against the embedded questions, not the paragraphs

        String queryString = "How many inhabitants live in Berlin?";

        System.out.println(magenta("-".repeat(100)));
        System.out.println(magenta("\nUSER QUESTION: ") + queryString);

        var embeddingModel = getEmbeddingModel("text-embedding-004");
        var embeddingStore = getEmbeddingStore();

        EmbeddingSearchResult<TextSegment> searchResults = embeddingStore.search(EmbeddingSearchRequest.builder()
            .maxResults(4)
            .minScore(0.7)
            .queryEmbedding(embeddingModel.embed(queryString).content())
            .build());

        ScoringModel scoringModel = getScoringModel();

        searchResults.matches().forEach(match -> {
            double score = scoringModel.score(match.embedded().metadata().getString(PARAGRAPH_KEY), queryString).content();

            System.out.println(yellow("\n-> Similarity: " + match.score() + " --- (Ranking score: " + score + ") ---\n") +
                "\n" + cyan("Embedded question: ") + match.embedded().text() +
                "\n" + cyan("  About paragraph: ") + match.embedded().metadata().getString(PARAGRAPH_KEY));
        });

        // =================================
        // Ask Gemini to generate a response

        ChatLanguageModel chatModel = getChatLanguageModel("gemini-1.5-pro-002");

        String concatenatedExtracts = searchResults.matches().stream()
            .map(match -> match.embedded().metadata().getString(PARAGRAPH_KEY))
            .distinct()
            .collect(Collectors.joining("\n---\n", "\n---\n", "\n---\n"));

        UserMessage userMessage = PromptTemplate.from("""
            You must answer the following question:
            
            {{question}}
            
            Base your answer on the following documentation extracts:
            
            {{extracts}}
            """).apply(Map.of(
            "question", queryString,
            "extracts", concatenatedExtracts
        )).toUserMessage();

        System.out.println(magenta("\nMODEL REQUEST:\n") + userMessage.text().replaceAll("\\n", "\n") + "\n");

        Response<AiMessage> response = chatModel.generate(userMessage);

        System.out.println(magenta("\nRESPONSE:\n") + response.content().text());
    }

    record QuestionParagraph(
        String question,
        TextSegment paragraph
    ) {
    }
}