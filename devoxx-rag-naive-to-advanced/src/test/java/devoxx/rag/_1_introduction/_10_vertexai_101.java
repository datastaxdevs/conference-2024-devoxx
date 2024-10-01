package devoxx.rag._1_introduction;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

/**
 * Not repeating concepts introduced better, illustrating mostly AbstracDevoxxSampleTest to remove the boilerplate.
 */
class _10_vertexai_101 extends AbstracDevoxxSampleTest {

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
