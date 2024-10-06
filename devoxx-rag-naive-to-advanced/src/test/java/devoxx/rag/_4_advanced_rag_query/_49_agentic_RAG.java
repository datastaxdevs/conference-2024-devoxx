package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.service.SystemMessage;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.*;
import static devoxx.rag._3_advanced_rag_ingestion._37_hypothetical_questions_embedding.getEmbeddingStore;

public class _49_agentic_RAG extends AbstractDevoxxTest {

    @Test
    public void agenticRAG() {

        AgenticAssistant assistant = AiServices.builder(AgenticAssistant.class)
            .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
            .tools(new HistoryGeographyTool())
            .build();

        String report = assistant.chat(
            "Write a report about the population of Berlin, its geographic situation, and its historical origins"
//            "Write a report about the cultural aspects of Berlin"
        );

        System.out.println(magenta("\n>>> FINAL RESPONSE REPORT:\n"));
        System.out.println(cyan(report));
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
                .contentRetriever(EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(getEmbeddingStore())
                    .embeddingModel(getEmbeddingModel(MODEL_EMBEDDING_TEXT))
                    .build())
                .build();

            Result<String> reportResult = topicAssistant.report(query);

            reportResult.sources().forEach(content -> {
                System.out.println(cyan("- Source: ") + content.textSegment().text());
            });

            System.out.println(yellow("\n-> Topic report: ") + reportResult.content().replaceAll("\\n", "\n"));

            return new TopicReport(query, reportResult.content());
        }
    }

    record TopicReport(String topic, String report) {}
}
