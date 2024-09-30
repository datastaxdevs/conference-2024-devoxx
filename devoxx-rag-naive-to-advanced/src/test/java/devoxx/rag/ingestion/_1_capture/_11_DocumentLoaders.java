package devoxx.rag.ingestion._1_capture;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URL;
import java.nio.file.Path;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _11_DocumentLoaders {

    @Test
    public void should_load_text_file() {
        String fileName = "/johnny.txt";
        URL fileURL = getClass().getResource(fileName);
        Path path = new File(fileURL.getFile()).toPath();
        Document myDoc = FileSystemDocumentLoader.loadDocument(path, new TextDocumentParser());
        myDoc.metadata().toMap().entrySet().forEach(entry -> {
            System.out.println(cyan(entry.getKey()) + " : " + entry.getValue());
        });
    }

    @Test
    public void should_load_url() {
        String url = "https://www.devoxx.com";
        Document myDoc = FileSystemDocumentLoader.loadDocument(url, new TextDocumentParser());
        myDoc.metadata().toMap().entrySet().forEach(entry -> {
            System.out.println(cyan(entry.getKey()) + " : " + entry.getValue());
        });
    }

    @Test
    public void should_load_custom() {}

}
