package devoxx.rag.guillaume;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;

import java.io.IOException;
import java.nio.file.Path;

public class _22_loading_pdf_documents {
    public static void main(String[] args) throws IOException {

        Document document = FileSystemDocumentLoader.loadDocument(
            Path.of("src/main/resources/cymbal-starlight-2024.pdf"),
            new ApachePdfBoxDocumentParser()
        );

        System.out.println("document = " + document.toTextSegment());
    }
}
