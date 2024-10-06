package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.*;
import static devoxx.rag._3_advanced_rag_ingestion._37_hypothetical_questions_embedding.getEmbeddingStore;

public class _49_agentic_RAG extends AbstractDevoxxTest {

    private static final EmbeddingModel EMBEDDING_MODEL = VertexAiEmbeddingModel.builder()
        .project(System.getenv("GCP_PROJECT_ID"))
        .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
        .location(System.getenv("GCP_LOCATION"))
        .publisher("google")
        .modelName("text-embedding-004")
        .maxSegmentsPerBatch(100)
        .autoTruncate(true)
        .outputDimensionality(768)
        .maxRetries(5)
        .build();

    @Test
    public void agenticRAG() {

        AgenticAssistant assistant = AiServices.builder(AgenticAssistant.class)
            .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
            .tools(new HistoryGeographyTool())
            .build();

        String report = assistant.chat(
            "Write a report about the population of Berlin, its geographic situation, and its historical origins");

        System.out.println(report);
    }

    interface AgenticAssistant {
        @SystemMessage("""
            You are a knowledgeable history and geography assistant.
            Your role is to write reports about a particular location or event,
            focusing on the key topics asked by the user.
            
            Think step by step:
            1) Identify the key topics the user is interested
            2) For each topic, devise a list of questions corresponding to those topics
            3) Search those questions in the database
            4) Collect all those answers together, and create the final report.
            """)
        String chat(String userMessage);
    }

    class HistoryGeographyTool extends AbstractDevoxxTest {

        private static final EmbeddingStore<TextSegment> EMBEDDING_STORE = getEmbeddingStore();

        /*
        @Tool("Search information in the database")
        List<String> searchInformationInDatabase(String query) {
            System.out.println(magenta(">>> Invoking `searchInformation` tool with query: ") + query);

            EmbeddingSearchResult<TextSegment> results = EMBEDDING_STORE.search(EmbeddingSearchRequest.builder()
                .maxResults(3)
                .minScore(0.8)
                .queryEmbedding(EMBEDDING_MODEL.embed(query).content())
                .build());

            List<String> output = new ArrayList<>();

            results.matches().forEach(match -> {
                String embeddedQuestion = match.embedded().text();
                String referenceParagraph = match.embedded().metadata().getString("paragraph");

                System.out.println(yellow("-> Question: ") + embeddedQuestion);

                output.add(String.format("""
                    # Question: %s

                    %s %n
                    """, embeddedQuestion, referenceParagraph));
            });

            return output;
        }
        */

        interface TopicAssistant {
            @SystemMessage("""
                You are a knowledgeable history and geography assistant who knows how to succinctly summarize a topic.
                Summarize the information for the topic asked by the user.
                """)
            Result<String> report(String subTopic);
        }

        @Tool("Search information in the database")
        TopicReport searchInformationInDatabase(String query) {
            System.out.println(magenta(">>> Invoking `searchInformation` tool with query: ") + query);

            TopicAssistant topicAssistant = AiServices.builder(TopicAssistant.class)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
                .contentRetriever(EmbeddingStoreContentRetriever.from(EMBEDDING_STORE))
                .build();

            Result<String> reportResult = topicAssistant.report(query);

            System.out.println(yellow("-> Topic eport: ") + reportResult.content().replaceAll("\\n", "\n"));

            return new TopicReport(query, reportResult.content());
        }
    }

    record TopicReport(String topic, String report) {}
}
