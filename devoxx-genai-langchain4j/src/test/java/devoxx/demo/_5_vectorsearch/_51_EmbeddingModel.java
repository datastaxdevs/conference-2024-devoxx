package devoxx.demo._5_vectorsearch;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import devoxx.demo.utils.AbstractDevoxxTestSupport;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;

@Slf4j
public class _51_EmbeddingModel extends AbstractDevoxxTestSupport {

    @Test
    public void testEmbeddingModel() {
        Response<Embedding> embedding = getEmbeddingModelGecko().embed("Hello, World!");
        log.info("Dimension: {}", embedding.content().dimension());
        log.info("Vector: {}", embedding.content().vectorAsList());
    }

}
