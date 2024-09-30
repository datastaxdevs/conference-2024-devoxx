package devoxx.rag.guillaume;

import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class _68_semantic_chunking {
    public static void main(String[] args) {
        String text = """
            Berlin is the capital and largest city of Germany, both by area and by population. \
            Its more than 3.85 million inhabitants make it the European Union's most populous \
            city, as measured by population within city limits. The city is also one of the \
            states of Germany, and is the third smallest state in the country in terms of area. \
            Berlin is surrounded by the state of Brandenburg, and Brandenburg's capital Potsdam \
            is nearby. The urban area of Berlin has a population of over 4.5 million and is \
            therefore the most populous urban area in Germany. The Berlin-Brandenburg capital \
            region has around 6.2 million inhabitants and is Germany's second-largest \
            metropolitan region after the Rhine-Ruhr region, and the sixth-biggest metropolitan \
            region by GDP in the European Union.
            
            Berlin was built along the banks of the Spree river, which flows into the Havel in \
            the western borough of Spandau. The city incorporates lakes in the western and \
            southeastern boroughs, the largest of which is Müggelsee. About one-third of the \
            city's area is composed of forests, parks and gardens, rivers, canals, and lakes.
            
            First documented in the 13th century and at the crossing of two important historic \
            trade routes, Berlin was designated the capital of the Margraviate of Brandenburg \
            (1417–1701), Kingdom of Prussia (1701–1918), German Empire (1871–1918), Weimar \
            Republic (1919–1933), and Nazi Germany (1933–1945). Berlin has served as a \
            scientific, artistic, and philosophical hub during the Age of Enlightenment, \
            Neoclassicism, and the German revolutions of 1848–1849. During the Gründerzeit, an \
            industrialization-induced economic boom triggered a rapid population increase in \
            Berlin. 1920s Berlin was the third-largest city in the world by population.
            
            After World War II and following Berlin's occupation, the city was split into West \
            Berlin and East Berlin, divided by the Berlin Wall. East Berlin was declared the \
            capital of East Germany, while Bonn became the West German capital. Following German \
            reunification in 1990, Berlin once again became the capital of all of Germany. Due \
            to its geographic location and history, Berlin has been called "the heart of Europe".
            
            The economy of Berlin is based on high tech and the service sector, encompassing a \
            diverse range of creative industries, startup companies, research facilities, and \
            media corporations.] Berlin serves as a continental hub for air and rail traffic and \
            has a complex public transportation network. Tourism in Berlin makes the city a \
            popular global destination. Significant industries include information technology, \
            the healthcare industry, biomedical engineering, biotechnology, the automotive \
            industry, and electronics.
            
            Berlin is home to several universities such as the Humboldt University of Berlin, \
            Technische Universität Berlin, the Berlin University of the Arts and the Free \
            University of Berlin. The Berlin Zoological Garden is the most visited zoo in \
            Europe. Babelsberg Studio is the world's first large-scale movie studio complex and \
            the list of films set in Berlin is long.
            
            Berlin is also home to three World Heritage Sites. Museum Island, the Palaces and \
            Parks of Potsdam and Berlin, and the Berlin Modernism Housing Estates. Other \
            landmarks include the Brandenburg Gate, the Reichstag building, Potsdamer Platz, \
            the Memorial to the Murdered Jews of Europe, and the Berlin Wall Memorial. Berlin \
            has numerous museums, galleries, and libraries.
            """;

        var embeddingModel = VertexAiEmbeddingModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("text-embedding-004")
            .publisher("google")
            .build();

        // split by sentences
        DocumentBySentenceSplitter splitter = new DocumentBySentenceSplitter(500, 0);
        List<String> sentences = Arrays.asList(splitter.split(text));

        // create groups of sentences (1 before, 2 after current sentence)
        List<List<String>> slidingWindowSentences = Utils.slidingWindow(sentences, 1, 2);
        List<TextSegment> concatenatedSentences = slidingWindowSentences.stream()
            .map(strings -> strings.stream()
                .collect(Collectors.joining(" ")))
            .map(TextSegment::from)
            .toList();

        // calculate vector embeddings for each of these sentence groups
        List<Embedding> embeddings = embeddingModel.embedAll(concatenatedSentences).content();

        // calculate the pair-wise similarities between each sentence groups
        List<Double> similarities = new ArrayList<>();
        for (int i = 0; i < embeddings.size() - 1; i++) {
            similarities.add(CosineSimilarity.between(embeddings.get(i), embeddings.get(i+1)));
        }

        // find the 5 lowest similiarities
        List<Double> lowestSimilarities = similarities.stream()
            .sorted()
            .limit(5)
            .toList();

        // find the 5 breakpoints (indices) where the similarity is the lowest 
        int[] lowestSimilaritiesIndices = lowestSimilarities.stream()
            .mapToInt(similarity ->
                IntStream.range(0, similarities.size())
                    .filter(streamIndex -> similarities.get(streamIndex).equals(similarity))
                    .findFirst()
                    .orElse(-1))
            .sorted()
            .toArray();

        System.out.println("Arrays.asList(lowestSimilaritiesIndices) = " + Arrays.toString(lowestSimilaritiesIndices));

        List<String> finalSentenceGroups = new ArrayList<>();

        int startIndex = 0;
        for (int i = 0; i < lowestSimilaritiesIndices.length; i++) {
            finalSentenceGroups.add(sentences.subList(startIndex, lowestSimilaritiesIndices[i])
                .stream().collect(Collectors.joining(" ")));
            startIndex = lowestSimilaritiesIndices[i];
        }
        finalSentenceGroups.add(sentences.subList(startIndex, sentences.size())
            .stream().collect(Collectors.joining(" ")));


        finalSentenceGroups.forEach(s -> {
            System.out.format("==========================%n %s %n", s);
        });
    }
}
