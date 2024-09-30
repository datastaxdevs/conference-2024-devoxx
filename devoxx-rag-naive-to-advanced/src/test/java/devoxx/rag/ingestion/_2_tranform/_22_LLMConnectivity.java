package devoxx.rag.ingestion._2_tranform;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstracDevoxxSampleTest;
import org.junit.jupiter.api.Test;

/**
 * Not repeating concepts introduced better, illustrating mostly AbstracDevoxxSampleTest to remove the boilerplate.
 */
class _22_LLMConnectivity extends AbstracDevoxxSampleTest {

    @Test
    public void should_reply_question() {
        String question = "What is the sky bue ?";
        Response<AiMessage> response =
                getChatLanguageModel(MODEL_GEMINI_PRO)
                        .generate(UserMessage.from(question));
        prettyPrint(response);
    }

    @Test
    public void should_reply_question_streaming() {
        String userMessage = "Give me a poem in 20 sentences about DEVOXX";
        getChatLanguageModelStreaming(MODEL_GEMINI_PRO)
                .generate(userMessage, new PrettyPrintStreamingResponseHandler());
    }




}
