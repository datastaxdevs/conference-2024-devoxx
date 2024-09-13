package devoxx.demo.gemini._1_vertexai;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.chat.TestStreamingResponseHandler;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiStreamingChatModel;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ID;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_LOCATION;

class _10_LanguageModelSayHello {

    @Test
    public void shouldSayHelloToLLM() {
        ChatLanguageModel chatLanguageModel= VertexAiGeminiChatModel.builder()
                .project(GCP_PROJECT_ID)
                .location(GCP_PROJECT_LOCATION)
                .modelName("gemini-pro")
                .build();
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(new UserMessage("Hi, tell ma joke "));

        Response<AiMessage> response = chatLanguageModel.generate(messages);
        System.out.println(response.content());
        System.out.println(response.finishReason());
        System.out.println(response.tokenUsage().inputTokenCount());
        System.out.println(response.tokenUsage().outputTokenCount());
        System.out.println(response.tokenUsage().totalTokenCount());
    }

    @Test
    public void shouldSayHelloToLLMStreaming() {

        StreamingChatLanguageModel model = VertexAiGeminiStreamingChatModel.builder()
                .project(GCP_PROJECT_ID)
                .location(GCP_PROJECT_LOCATION)
                .modelName("gemini-pro")
                .build();

        String userMessage = "What is the capital of Germany?";

        // when
        TestStreamingResponseHandler<AiMessage> handler = new TestStreamingResponseHandler<>();
        model.generate(userMessage, handler);
        Response<AiMessage> response = handler.get();

    }


}
