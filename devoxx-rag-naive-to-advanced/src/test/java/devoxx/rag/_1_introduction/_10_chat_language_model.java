package devoxx.rag._1_introduction;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

class _10_chat_language_model extends AbstractDevoxxTest {

    @Test
    public void should_chat_language_model() {
        System.out.println(yellow("Using Chat Model:"));
        ChatLanguageModel chatModel = VertexAiGeminiChatModel.builder()
                .project(System.getenv("GCP_PROJECT_ID"))
                .location(System.getenv("GCP_LOCATION"))
                .modelName("gemini-pro")
                .build();

        String question = "What is the sky bue ?";
        System.out.println(cyan("Question: ") + question);

        Response<AiMessage> response = chatModel.generate(UserMessage.from( question));
        prettyPrint(response);
    }

    @Test
    public void should_tune_llm_request() {
        System.out.println(yellow("Tuning Chat Model:"));
        ChatLanguageModel chatModel = VertexAiGeminiChatModel.builder()
                .project(System.getenv("GCP_PROJECT_ID"))
                .location(System.getenv("GCP_LOCATION"))
                .modelName("gemini-pro")

                // TUNING MODEL
                .temperature(0.7f)
                .topK(3)
                .topP(.8f)
                .maxRetries(3)
                .maxOutputTokens(2000)
                // <---

                .build();
        String question = "What are the profiles of Devoxx attendees";
        Response<AiMessage> response = chatModel.generate(UserMessage.from( question));
        prettyPrint(response);
    }


    /** Streaming handler to log results. */
    public static class PrettyPrintStreamingResponseHandler
            implements StreamingResponseHandler<AiMessage> {
        @Override
        public void onNext(String s) { System.out.println(s); }
        @Override
        public void onComplete(Response<AiMessage> response) {  prettyPrint(response);}
        @Override
        public void onError(Throwable throwable) { System.out.println("Error : " + throwable.getMessage());}
    }

    @Test
    public void should_chat_language_model_streaming() {
        System.out.println(yellow("Using Chat Model (Streaming):"));
        String question = "Give me a poem in 20 sentences about DEVOXX";
        System.out.println(cyan("Question: ") + question);
        getChatLanguageModelStreaming(MODEL_GEMINI_PRO).generate(question,new PrettyPrintStreamingResponseHandler());
    }

}
