package devoxx.rag.guillaume;

import com.google.cloud.vertexai.api.Schema;
import com.google.cloud.vertexai.api.Type;
import com.google.gson.Gson;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;

import java.util.List;
import java.util.stream.Stream;

public class _63_hypothetical_questions_embedding {
    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        String text = """
            Dimensionality Reduction: Simplifying Complex Data
            
            Dimensionality reduction is a technique used to simplify complex datasets by reducing the number of features (or dimensions) while preserving as much relevant information as possible. 
            This process can be beneficial for several reasons:
            
            Computational Efficiency: High-dimensional data can be computationally expensive to process. By reducing dimensionality, you can significantly speed up algorithms and improve model performance.
            Visualization: Visualizing high-dimensional data is challenging. Dimensionality reduction techniques can help to create more interpretable visualizations, making it easier to understand relationships and patterns within the data.
            Noise Reduction: Redundant or irrelevant features can introduce noise into your data. Dimensionality reduction can help to eliminate or reduce the impact of noise.
            Feature Engineering: By combining or transforming features, dimensionality reduction can create new, more informative features that can improve model performance.
            
            Common Dimensionality Reduction Techniques
            
            Principal Component Analysis (PCA): This is a popular technique that finds a new set of uncorrelated variables (principal components) that capture the most variance in the data.
            Linear Discriminant Analysis (LDA): LDA is a supervised technique that seeks to find a projection that maximizes the separation between classes.
            t-SNE: t-SNE is a nonlinear technique that is particularly effective for visualizing high-dimensional data in low-dimensional spaces.
            Autoencoders: Autoencoders are neural networks that learn to compress and reconstruct data. They can be used for dimensionality reduction by training them to learn a lower-dimensional representation of the input data.
            
            When to Use Dimensionality Reduction:
            
            High-dimensional data: When you have a large number of features.
            Computational efficiency: When you need to improve the speed of your algorithms.
            Visualization: When you want to visualize complex data.
            Noise reduction: When you suspect your data contains noise.
            Feature engineering: When you want to create new, more informative features.
            
            Choosing the Right Technique:
            
            The best dimensionality reduction technique depends on your specific use case and the characteristics of your data. Consider factors such as the number of dimensions, the nature of the data (linear vs. nonlinear), and whether you have labeled data.
            
            By understanding dimensionality reduction and applying the appropriate techniques, you can simplify complex datasets, improve model performance, and gain valuable insights from your data.
            """;

        VertexAiGeminiChatModel gemini = VertexAiGeminiChatModel.builder()
            .project(System.getenv("GCP_PROJECT_ID"))
            .location(System.getenv("GCP_LOCATION"))
            .modelName("gemini-1.5-flash-002")
            .responseSchema(Schema.newBuilder()
                .setType(Type.ARRAY)
                .setItems(Schema.newBuilder().setType(Type.STRING).build())
                .build())
            .build();


        DocumentByParagraphSplitter splitter = new DocumentByParagraphSplitter(2000, 0);

        List<TextSegment> sentences = Stream.of(splitter.split(text))
            .map(sentence -> TextSegment.from(sentence))
            .toList();

        sentences.forEach(textSegment -> {
            System.out.println("\n=====================================\n" + textSegment.text());

            Response<AiMessage> aiResult = gemini.generate(List.of(
                SystemMessage.from("Suggest 5 questions whose answer could be the following text:"),
                UserMessage.from(textSegment.text())
            ));
            String[] questions = gson.fromJson(aiResult.content().text(), String[].class);

            System.out.println("\nQUESTIONS:\n");
            for (int i = 0; i < questions.length; i++) {
                String question = questions[i];
                System.out.println((i+1) + ") " + question);
            }

        });


    }
}