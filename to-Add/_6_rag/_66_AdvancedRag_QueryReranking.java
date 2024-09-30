package devoxx.rag._6_rag;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.cohere.CohereScoringModel;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.service.AiServices;
import devoxx.rag.TestSupport;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;


/**
 * ReRanking
 * https://dashboard.cohere.com/welcome/register
 */
public class _66_AdvancedRag_QueryReranking extends TestSupport {

    @Test
    public void shouldRerankResult() {

        // Re Ranking
        ScoringModel scoringModel = CohereScoringModel.withApiKey(System.getenv("COHERE_API_KEY"));

        ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .minScore(0.8)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(createRetriever("/johnny.txt"))
                .contentAggregator(contentAggregator)
                .build();

        Assistant assistant =  AiServices.builder(Assistant.class)
                .chatLanguageModel(getChatLanguageModelChatBison())
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        System.out.println( assistant.answer("Tell me 10 things about Johnny"));
    }
}