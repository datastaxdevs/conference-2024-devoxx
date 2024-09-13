package devoxx.demo._1_vertexai;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiImageModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ENDPOINT;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_ID;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_LOCATION;
import static devoxx.demo.devoxx.Utilities.GCP_PROJECT_PUBLISHER;
import static org.assertj.core.api.Assertions.assertThat;

class _14_ImageModel_GenerateTest {

    @Test
    public void shouldSayHelloToImageModel() {
        ImageModel imagenModel = VertexAiImageModel.builder()
                .project(GCP_PROJECT_ID)
                .endpoint(GCP_PROJECT_ENDPOINT)
                .location(GCP_PROJECT_LOCATION)
                .publisher(GCP_PROJECT_PUBLISHER)
                .modelName("imagegeneration@005")
                .maxRetries(2)
                .withPersisting()
                .build();

        Response<List<Image>> imageListResponse = imagenModel.generate("photo of a sunset over Malibu beach", 3);
        assertThat(imageListResponse.content()).hasSize(3);
        imageListResponse.content().forEach(img -> {
            assertThat(img.url()).isNotNull();
            assertThat(img.base64Data()).isNotNull();
            System.out.println(img.url());
        });

    }
}
