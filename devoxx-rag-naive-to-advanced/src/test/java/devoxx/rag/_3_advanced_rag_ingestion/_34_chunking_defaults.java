package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentByCharacterSplitter;
import dev.langchain4j.data.document.splitter.DocumentByLineSplitter;
import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

/**
 * CHUNKING DOCUMENTS
 */
public class _34_chunking_defaults extends AbstractDevoxxTest {

    @Test
    public void should_chunk_document_recursive() {
        // Parse Document
        Document document = loadDocumentText("text/johnny.txt");
        System.out.println(cyan("ORIGINAL TEXT"));
        System.out.println(document.text().trim());

        // QUICK RECURSIVE
        System.out.println(cyan("\nRECURSIVE SPLITTER"));
        List<TextSegment> chunks1 = DocumentSplitters.recursive(300, 20).split(document);
        for (TextSegment chunk : chunks1) {
            System.out.println("-> " + chunk.text().replaceAll("\\n", "\\n"));
        }

        System.out.println(cyan("\nSPLIT BY CHARACTERS"));
        List<TextSegment> chunks2 = new DocumentByCharacterSplitter(300, 20).split(document);
        for (TextSegment chunk : chunks2) {
            System.out.println("-> " + chunk.text());
        }

        System.out.println(cyan("\nSPLIT BY LINES"));
        List<TextSegment> chunks3 = new DocumentByLineSplitter(300, 20).split(document);
        for (TextSegment chunk : chunks3) {
            System.out.println("-> " + chunk.text());
        }

        System.out.println(cyan("\nSPLIT BY SENTENCES"));
        List<TextSegment> chunks4 = new DocumentBySentenceSplitter(300, 20).split(document);
        for (TextSegment chunk : chunks4) {
            System.out.println("-> " + chunk.text());
        }

    }
}
