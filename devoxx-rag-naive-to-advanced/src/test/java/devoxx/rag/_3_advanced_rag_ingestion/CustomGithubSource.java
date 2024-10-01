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
public class CustomGithubSource implements DocumentSource {

    private final String githubAccount;
    private final String githubRepo;
    private final String branch;
    private final String filename;

    public String getFullUrl() {
        return "https://raw.githubusercontent.com/" +
                githubAccount + "/" +
                githubRepo + "/" +
                branch + "/" +
                filename;
    }

    @Override
    public InputStream inputStream() throws IOException {
        try {
            return new URI(getFullUrl().toString()).toURL().openConnection().getInputStream();
        } catch (URISyntaxException e) {
            throw new IOException(e);
        }
    }

    @Override
    public Metadata metadata() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("account", githubAccount);
        metadata.put("repo", githubRepo);
        metadata.put("branch", branch);
        metadata.put("file", filename);
        return Metadata.from(metadata);
    }
}
