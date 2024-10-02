package devoxx.rag._1_introduction;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiImageModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

public class _10_image_model {

    @Test
    public void shouldSayHelloToImageModel() {
        ImageModel imagenModel = VertexAiImageModel.builder()
                .project(System.getenv("GCP_PROJECT_ID"))
                .location(System.getenv("GCP_LOCATION"))
                .endpoint(System.getenv("GCP_VERTEXAI_ENDPOINT"))
                .publisher("google")
                .modelName("imagegeneration@005")
                .maxRetries(2)
                .withPersisting()
                .build();

        Response<List<Image>> imageListResponse = imagenModel
                .generate("photo of a sunset over Malibu beach", 3);
        imageListResponse.content().forEach(img -> {
            assertThat(img.url()).isNotNull();
            assertThat(img.base64Data()).isNotNull();
            System.out.println(img.url());
        });

    }
}
