package devoxx.rag.experiments;

import com.google.cloud.vertexai.api.Schema;
import com.google.cloud.vertexai.api.Type;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {

        VertexAiGeminiChatModel model = VertexAiGeminiChatModel.builder()
            .project("genai-java-demos")
            .location("europe-west1")
            .modelName("gemini-1.5-flash")
            .responseSchema(Schema.newBuilder()
                .setType(Type.ARRAY)
                .setItems(Schema.newBuilder().setType(Type.STRING).build())
                .build())
            .build();

        String response = model.generate("""
            Split the text below into a list of sentences.
            ---
            At the core of RAG lies the concept of "chunking" — a process essential for breaking down large volumes of data into manageable segments. 
            Chunking ensures that LLMs receive coherent and digestible information, preventing them from being overwhelmed by extensive documents or datasets. 
            This is particularly advantageous when dealing with vast amounts of data scattered across multiple sources.
            
            Despite the impressive capabilities of modern LLMs, processing entire documents or massive datasets in one go can be resource-intensive and time-consuming. 
            Here's where chunking comes to the rescue, dividing the data into smaller, more manageable pieces that align with the LLM's processing capacity, thereby optimizing resource utilization.
            """);

        System.out.println(response);
    }
}