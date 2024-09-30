package devoxx.rag.guillaume;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;

import java.nio.file.Path;

public class _50_splitting_by_sentence {
    public static void main(String[] args) {
        Document document = FileSystemDocumentLoader.loadDocument(
            Path.of("src/main/resources/cymbal-starlight-2024.txt"));

        new DocumentBySentenceSplitter(100, 10)
            .split(document)
            .forEach(textSegment -> {
                System.out.println("---------\n" + textSegment.text());
            });
    }
}