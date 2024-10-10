package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

public class _48_2_sample_function_calling_weather extends AbstractDevoxxTest {

    record WeatherForecast(String location, String forecast, int temperature) {}

    static class WeatherForecastService {
        @Tool("Get the weather forecast and temperature for a location")
        WeatherForecast getForecast(@P("Location to get the forecast for") String location) {
            if (location.equals("Paris")) {
                return new WeatherForecast("Paris", "sunny", 20);
            } else if (location.equals("Antwerp")) {
                return new WeatherForecast("Antwerp", "warm", 32);
            } else {
                return new WeatherForecast("Unknown", "unknown", 0);
            }
        }
    }

    interface WeatherAssistant {
        String weatherForecastQuery(String userMessage);
    }

    @Test
    public void testWeatherFunctionCall() {
        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName(MODEL_GEMINI_FLASH)
            .build();

        WeatherForecastService weatherForecastService = new WeatherForecastService();

        WeatherAssistant assistant = AiServices.builder(WeatherAssistant.class)
            .chatLanguageModel(model)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .tools(weatherForecastService)
            .build();

        System.out.println(assistant.weatherForecastQuery(
            "Is it warmer in Antwerp or in Paris?"));
    }
}
