package devoxx.rag._2_ingestion._1_capture;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.source.FileSystemSource;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;

public class _12_DocumentParsers {

    @Test
    public void should_load_text_file() throws IOException {
        try(InputStream data = FileSystemSource.from("johny.txt").inputStream()) {
            Document langchainDocument = new TextDocumentParser().parse(data);

            System.out.println(langchainDocument.metadata());
        }
    }

    @Test
    public void should_load_pdf_file() {}

    @Test
    public void should_load_html_file() {}
}
