package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.stream.Collectors;

import static com.datastax.astra.internal.utils.AnsiUtils.*;

public class _38_contextual_retrieval extends AbstractDevoxxTest {

    public static final String ORIGINAL = "original";

    @Test
    public void contextualRetrieval() {

        Document document = loadDocumentText("text/berlin.txt");

        var gemini = getChatLanguageModel(MODEL_GEMINI_PRO);
        var embeddingModel = getEmbeddingModel(MODEL_EMBEDDING_TEXT);

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        PromptTemplate promptTemplate = PromptTemplate.from("""
            <document>
            {{wholeDocument}}
            </document>
            Here is the chunk we want to situate within the whole document
            <chunk>
            {{chunk}}
            </chunk>
            Please give a short succinct context to situate this chunk within the overall document \
            for the purposes of improving search retrieval of the chunk. \
            Answer only with the succinct context and nothing else.
            """);

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(DocumentSplitters.recursive(1000, 0))
            .embeddingStore(embeddingStore)
            .embeddingModel(embeddingModel)
            .textSegmentTransformer(segment -> {
                Response<AiMessage> generatedChunk = gemini.generate(promptTemplate.apply(Map.of(
                        "chunk", segment.text(),
                        "wholeDocument", document.text()))
                    .toUserMessage());

                System.out.println("\n" + "-".repeat(100));
                System.out.println(yellow("ORIGINAL:\n") + segment.text());
                System.out.println(yellow("\nCONTEXTUALIZED CHUNK:\n") + generatedChunk.content().text());

                return TextSegment.from(generatedChunk.content().text(), new Metadata().put(ORIGINAL, segment.text()));
            })
            .build();
        ingestor.ingest(document);

        String queryString = "What's the population of Berlin?'";

        System.out.println("=".repeat(100) + cyan("\nQUESTION: ") + queryString);

        ScoringModel scoringModel = getScoringModel();

        EmbeddingSearchResult<TextSegment> results = embeddingStore.search(EmbeddingSearchRequest.builder()
            .minScore(0.7)
            .maxResults(5)
            .queryEmbedding(embeddingModel.embed(queryString).content())
            .build());

        results.matches().forEach(match -> {
            double score = scoringModel.score(match.embedded().text(), queryString).content();

            System.out.println(magenta("\n-> Similarity: " + match.score() + " --- (Ranking score: " + score + ") ---\n"));
            System.out.println(yellow("ORIGINAL:\n") + match.embedded().metadata().getString(ORIGINAL));
            System.out.println(yellow("\nCONTEXTUALIZED CHUNK:\n") + match.embedded().text());

        });


        // =================================
        // Ask Gemini to generate a response

        ChatLanguageModel chatModel = getChatLanguageModel(MODEL_GEMINI_PRO);

        String concatenatedExtracts = results.matches().stream()
            .map(match -> match.embedded().metadata().getString(ORIGINAL))
            .filter(original -> scoringModel.score(original, queryString).content() > 0.7)
            .distinct()
            .collect(Collectors.joining("\n---\n", "\n---\n", "\n---\n"));

        UserMessage userMessage = PromptTemplate.from("""
            You must answer the following query:
            
            {{query}}
            
            Base your answer on the following documentation extracts:
            
            {{extracts}}
            """).apply(Map.of(
            "query", queryString,
            "extracts", concatenatedExtracts
        )).toUserMessage();

        System.out.println(magenta("\nMODEL REQUEST:\n") + userMessage.text().replaceAll("\\n", "\n") + "\n");

        Response<AiMessage> response = chatModel.generate(userMessage);

        System.out.println(magenta("\nRESPONSE:\n") + response.content().text());

    }
}