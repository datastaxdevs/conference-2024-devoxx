package devoxx.rag.guillaume;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentByRegexSplitter;

public class _52_splitting_regex {
    public static void main(String[] args) {
        Document document = Document.from("""
            # Big title
            Some intro
            ## Chapter One
            More text
            ### Section One
            Some more text.
            """);

        DocumentByRegexSplitter splitter = new DocumentByRegexSplitter("# ", "# ", 50, 10,
            new DocumentByRegexSplitter("## ", "## ", 50, 10,
                new DocumentByRegexSplitter("### ", "### ", 50, 10)));

        splitter
            .split(document)
            .forEach(textSegment -> {
                System.out.println("---------\n" + textSegment.text());
            });
    }
}