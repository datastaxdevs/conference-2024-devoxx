package devoxx.demo._1_vertexai;

import dev.langchain4j.model.language.LanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiLanguageModel;
import org.junit.jupiter.api.Test;

import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ENDPOINT;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ID;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_LOCATION;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_PUBLISHER;

class _11_LanguageModel_SayHelloTest {

    @Test
    public void shouldSayHelloToLLM() {
        LanguageModel llm = VertexAiLanguageModel.builder()
                .publisher(GCP_PROJECT_PUBLISHER)
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)
                .publisher(GCP_PROJECT_PUBLISHER)
                .modelName("text-bison")
                .build();
        Response<String> response = llm.generate("Hello, LLM!");
        System.out.println(response.content());
    }


}
