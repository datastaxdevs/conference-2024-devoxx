package devoxx.rag.guillaume;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;

import java.nio.file.Path;

public class _35_multimodal_question {
    public static void main(String[] args) {

        var model = VertexAiGeminiChatModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("gemini-1.5-flash-002")
            .temperature(0.7f)
            .topP(0.8f)
            .maxOutputTokens(1000)
            .build();

        Response<AiMessage> response = model.generate(UserMessage.from(
            PdfFileContent.from(Path.of("src/main/resources/cymbal-starlight-2024.pdf").toUri()),
            TextContent.from("What is the cargo capacity of Cymbal Starlight?")
        ));

        System.out.println("response = " + response.content().text());
    }
}
