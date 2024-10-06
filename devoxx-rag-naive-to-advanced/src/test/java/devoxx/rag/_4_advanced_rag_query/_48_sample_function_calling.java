package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class _48_sample_function_calling extends AbstractDevoxxTest {

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
            .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .tools(calculator)
            .build();

        String answer = assistant.chat("How much is 754 + 926?");
        System.out.println(answer);
    }

    @Test
    public void low_level_tools_example() {

        List<ToolSpecification> toolSpecifications = ToolSpecifications.toolSpecificationsFrom(new Calculator());

        UserMessage userMessage = new UserMessage("How much is 754 + 926?");

        Response<AiMessage> response = getChatLanguageModel(MODEL_GEMINI_PRO)
            .generate(singletonList(userMessage), toolSpecifications);

        AiMessage aiMessage = response.content();
        assertThat(aiMessage.hasToolExecutionRequests()).isTrue();
        assertThat(aiMessage.toolExecutionRequests()).hasSize(1);

        ToolExecutionRequest toolExecutionRequest = aiMessage.toolExecutionRequests().get(0);
        assertThat(toolExecutionRequest.name()).isEqualTo("add");
        assertThat(toolExecutionRequest.arguments())
            .isEqualToIgnoringWhitespace("{\"arg1\":926.0,\"arg0\":754.0}");
    }

}
