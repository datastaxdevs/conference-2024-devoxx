package devoxx.rag.utils;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import devoxx.rag.TestSupport;
import org.junit.jupiter.api.Test;

import java.util.List;

import static devoxx.rag.devoxx.Utilities.GCP_PROJECT_ID;
import static devoxx.rag.devoxx.Utilities.GCP_PROJECT_LOCATION;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class _71_CallFunctionTest extends TestSupport {

    // Get the model
    ChatLanguageModel model= VertexAiGeminiChatModel.builder()
            .project(GCP_PROJECT_ID)
            .location(GCP_PROJECT_LOCATION)
            .modelName("gemini-pro")
            .build();

    static class Calculator {
        @Tool("Adds two given numbers")
        double add(double a, double b) {
            System.out.printf("Called add(%s, %s)%n", a, b);
            return a + b;
        }
    }

    interface Assistant {
        String chat(String userMessage);
    }

    @Test
    public void testFunctionCalling1() {



        Calculator calculator = new Calculator();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .tools(calculator)
                .build();

        String answer = assistant.chat("How much is 754 + 926?");
        System.out.println(answer);
    }

    @Test
    void Low_level_Tools_Example() {

        List<ToolSpecification> toolSpecifications = ToolSpecifications.toolSpecificationsFrom(new Calculator());

        UserMessage userMessage = new UserMessage("How much is 754 + 926?");

        Response<AiMessage> response = model.generate(singletonList(userMessage), toolSpecifications);

        AiMessage aiMessage = response.content();
        Assertions.assertThat(aiMessage.hasToolExecutionRequests()).isTrue();
        Assertions.assertThat(aiMessage.toolExecutionRequests()).hasSize(1);

        ToolExecutionRequest toolExecutionRequest = aiMessage.toolExecutionRequests().get(0);
        Assertions.assertThat(toolExecutionRequest.name()).isEqualTo("add");
        Assertions.assertThat(toolExecutionRequest.arguments()).isEqualToIgnoringWhitespace("{\"arg1\":926.0,\"arg0\":754.0}");
    }

}
