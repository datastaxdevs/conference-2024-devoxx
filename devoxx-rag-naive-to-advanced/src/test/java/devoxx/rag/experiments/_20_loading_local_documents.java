package devoxx.rag.experiments;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.TextDocumentParser;

import java.io.IOException;
import java.io.InputStream;

public class _20_loading_local_documents {
    public static void main(String[] args) throws IOException {

        InputStream resourceStream = _20_loading_local_documents.class.getResourceAsStream(
            "/cymbal-starlight-2024.txt");

        TextDocumentParser parser = new TextDocumentParser();
        Document document = parser.parse(resourceStream);

        System.out.println("document = " + document.toTextSegment());
    }
}
