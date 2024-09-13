package devoxx.demo._1_vertexai;

import dev.langchain4j.model.language.LanguageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiLanguageModel;
import org.junit.jupiter.api.Test;

import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ENDPOINT;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ID;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_LOCATION;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_PUBLISHER;

public class _12_LanguageModel_ModelTuningTest {

    @Test
    public void shouldFineTuneYourRequest() {

        LanguageModel llm =  VertexAiLanguageModel.builder()
                .publisher(GCP_PROJECT_PUBLISHER)
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)

                .modelName("text-bison")
                .temperature(0.7)
                .topK(3)
                .topP(.8) // no both at same time
                .maxRetries(3)
                .maxOutputTokens(2000)

                .build();
        Response<String> response = llm.generate("What it the capital of France?");
        System.out.println(response.content());
    }
}
