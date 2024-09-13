package devoxx.demo._2_chunking;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentByCharacterSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

public class _21_ChunkingTechniques {

    @Test
    public void defaultChunking() {
        // Parse Document
        Path path = new File(Objects.requireNonNull(getClass().getResource("/johnny.txt")).getFile()).toPath();
        Document document = FileSystemDocumentLoader.loadDocument(path, new TextDocumentParser());

        System.out.println("ORIGINAL TEXT");
        System.out.println(document.text().trim());

        // QUICK RECURSIVE
        List<TextSegment> chunks1 = DocumentSplitters.recursive(300, 20).split(document);
        for (TextSegment chunk : chunks1) {
            System.out.println("-" + chunk.text());
        }

        //WORD
        System.out.println("-----------------------");
        List<TextSegment> chunks2 = new DocumentByCharacterSplitter(300, 20).split(document);
        for (TextSegment chunk : chunks2) {
            System.out.println("-" + chunk.text());
        }

    }
}
