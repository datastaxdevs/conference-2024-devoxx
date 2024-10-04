package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.DocumentSource;
import dev.langchain4j.data.document.Metadata;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

@Data @AllArgsConstructor
public class HugginFaceDatasetSource implements DocumentSource {

    private final String account_name;
    private final String dataset_name;
    private final String filename;

    public String getFullUrl() {
        return "https://huggingface.co/datasets/" + account_name + "/" + dataset_name + "/resolve/main/" + filename;
    }

    @Override
    public InputStream inputStream() throws IOException {
        try {
            return new URI(getFullUrl()).toURL().openConnection().getInputStream();
        } catch (URISyntaxException e) {
            throw new IOException(e);
        }
    }

    @Override
    public Metadata metadata() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("account_name", account_name);
        metadata.put("dataset_name", dataset_name);
        metadata.put("filename", filename);
        return Metadata.from(metadata);
    }
}
