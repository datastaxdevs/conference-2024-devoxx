package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentLoader;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.parser.apache.poi.ApachePoiDocumentParser;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _30_loader_and_parsers extends AbstractDevoxxTest {

    /**
     * FileSystemDocumentLoader + PARSER => DOCUMENT
     */
    @Test
    public void should_load_from_fileSystem()  throws URISyntaxException {

        Document doc1 = parseFileSystemDocument("/text/johnny.txt");
        prettyPrint(doc1);

        Document doc2 = parseFileSystemDocument("/pdf/devoxx.pdf");
        prettyPrint(doc2);

        Document doc3 = parseFileSystemDocument("/doc/devoxx.docx");
        prettyPrint(doc3);
    }

    /**
     * URL Document Loader + Parser => Document
     */
    @Test
    public void should_parse_url() throws URISyntaxException {
        String url = "https://www.devoxx.com";
        DocumentParser parser = selectParser(url);
        Document doc = UrlDocumentLoader.load("https://www.devoxx.com", parser);
        prettyPrint(doc);
    }

    /**
     * Customer Document Loader + Parser => Document
     */
    @Test
    public void should_parse_custom_sources() {
        // Github
        CustomGithubSource source = new CustomGithubSource("datastaxdevs",
                "conference-2024-devoxx", "main",
                "devoxx-rag-naive-to-advanced/src/test/resources/pdf/devoxx.pdf");
        DocumentParser parser = selectParser(source.getFullUrl());
        Document githubFile = DocumentLoader.load(source, parser);
        prettyPrint(githubFile);

        // Hugging Face DataSet
        HugginFaceDatasetSource hugginFaceDatasetSource =
                new HugginFaceDatasetSource("datastax",
                        "philosopher-quotes",
                        "philosopher-quotes.csv");
        DocumentParser parser2 = selectParser(hugginFaceDatasetSource.getFullUrl());
        Document hfDataset = DocumentLoader.load(hugginFaceDatasetSource, parser2);
        prettyPrint(hfDataset);
    }


    /**
     * Get the proper parser based on the file extension.
     */
    private DocumentParser selectParser(String fileName) {
        String extension = fileName.substring(fileName.lastIndexOf(".") + 1);
        DocumentParser parser = null;
        switch (extension) {
            case "pdf" -> parser = new ApachePdfBoxDocumentParser();
            case "doc","docx","ppt","pptx","xlsx","xls" -> parser = new ApachePoiDocumentParser();
            case "html","txt,json,csv" -> parser = new TextDocumentParser();
            default -> parser = new TextDocumentParser();
        }
        System.out.println(cyan("Parser: ") + parser.getClass().getSimpleName());
        return parser;
    }

    /**
     * Create a document from a file
     */
    private Document parseFileSystemDocument(String fileName) throws URISyntaxException {
        DocumentParser parser = selectParser(fileName);
        Path path = Paths.get(Objects.requireNonNull(getClass().getResource(fileName)).toURI());
        return FileSystemDocumentLoader.loadDocument(path, parser);
    }

    /**
     * Take a document and pretty print it
     */
    private void prettyPrint(Document doc) {
        System.out.println(yellow("metadata") + " :");
        doc.metadata().toMap().entrySet().forEach(entry -> {
            System.out.println(cyan(entry.getKey()) + " : " + entry.getValue());
        });
        System.out.println("\n" + yellow("content") + " :\n" + doc.text().replaceAll("\\n\\n", " "));
        System.out.println("------------------------------------\n");
    }

}
