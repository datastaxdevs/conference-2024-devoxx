package devoxx.rag.experiments;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;

import java.nio.file.Path;

public class _51_splitting_recursive {
    public static void main(String[] args) {
        Document document = FileSystemDocumentLoader.loadDocument(
            Path.of("src/main/resources/cymbal-starlight-2024.txt"));

        DocumentSplitters.recursive(500, 100)
            .split(document)
            .forEach(textSegment -> {
                System.out.println("---------\n" + textSegment.text());
            });
    }
}