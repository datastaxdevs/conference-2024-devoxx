package devoxx.rag._2_ingestion._1_capture;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentLoader;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSource;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.parser.apache.poi.ApachePoiDocumentParser;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _11_DocumentLoaders {

    /**
     * Pick a parser
     *
     * @param fileName
     *      filename
     * @return
     */
    private DocumentParser getParser(String fileName) {
        String extension = fileName.substring(fileName.lastIndexOf(".") + 1);
        DocumentParser parser = null;
        switch (extension) {
            case "pdf" -> parser = new ApachePdfBoxDocumentParser();
            case "doc","docx","ppt","pptx","xlsx","xls" -> parser = new ApachePoiDocumentParser();
            case "html","txt" -> parser = new TextDocumentParser();
            default -> throw new IllegalArgumentException("Unsupported document format: " + extension);
        }
        return parser;
    }

    private void parseDocument(String fileName) {

        // Load file
        URL fileURL = getClass().getResource(fileName);
        Path path = new File(fileURL.getFile()).toPath();

        // Pick parser
        DocumentParser parser = getParser(fileName);
        System.out.println(cyan("Parser: ") + parser.getClass().getSimpleName());

        // Parse and log
        Document myDoc = FileSystemDocumentLoader.loadDocument(path, parser);
        myDoc.metadata().toMap().entrySet().forEach(entry -> {
            System.out.println(cyan(entry.getKey()) + " : " + entry.getValue());
        });
    }

    @Test
    public void should_parse_local_files() {

        parseDocument("/text/johnny.txt");

        parseDocument("/pdf/devoxx.pdf");

        parseDocument("/doc/devoxx.docx");
    }

    @Test
    public void should_load_url() {
        String url = "https://www.devoxx.com";
        Document myDoc = UrlDocumentLoader.load(url, new TextDocumentParser());
        myDoc.metadata().toMap().entrySet().forEach(entry -> {
            System.out.println(cyan(entry.getKey()) + " : " + entry.getValue());
        });
        System.out.println(cyan("content") + " : " + myDoc.text());
    }

    @Test
    public void should_load_custom() {

        DocumentSource gcsSource = new DocumentSource() {

            @Override
            public InputStream inputStream() throws IOException {
                // Load from a custom source
                // GCS :)
                return null;
            }

            @Override
            public Metadata metadata() {
                return Metadata.metadata("title", "My custom document")
                        .put("author", "Me");
            }
        };


    }


}
