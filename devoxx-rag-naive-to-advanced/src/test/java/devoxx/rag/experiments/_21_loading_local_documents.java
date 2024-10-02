package devoxx.rag.experiments;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;

import java.io.IOException;
import java.nio.file.Path;

public class _21_loading_local_documents {
    public static void main(String[] args) throws IOException {

        Document document = FileSystemDocumentLoader.loadDocument(
            Path.of("src/main/resources/cymbal-starlight-2024.txt"));

        System.out.println("document = " + document.toTextSegment());
    }
}
