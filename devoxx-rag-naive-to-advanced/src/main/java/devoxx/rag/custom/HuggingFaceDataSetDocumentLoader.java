package devoxx.rag.custom;

import dev.langchain4j.data.document.DocumentLoader;
import dev.langchain4j.data.document.DocumentSource;
import dev.langchain4j.data.document.Metadata;

import java.io.IOException;
import java.io.InputStream;

public class HuggingFaceDataSetDocumentLoader implements DocumentSource {

    //private static final String hugginFaceToken;

    //private static final String accountName;

    //private static final String datasetName;

    //private static final String fileName;

    public HuggingFaceDataSetDocumentLoader() {

    }

    @Override
    public InputStream inputStream() throws IOException {
        return null;
    }

    @Override
    public Metadata metadata() {
        return null;
    }
}
