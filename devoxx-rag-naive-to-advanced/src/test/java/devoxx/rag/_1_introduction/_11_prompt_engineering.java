package devoxx.rag._1_introduction;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

public class _11_prompt_engineering extends AbstractDevoxxTest {

    @Test
    public void should_use_prompt_template() {

        PromptTemplate promptTemplate = PromptTemplate.from("""
            ----
            DIRECTIVE:
            - You are an assistant targeting Java developers
            - Your mission is to provide helpful answers
            ----
            CONSTRAINTS
            - If you do not know, say you do not know
            - Only answers related to java programming languages
            ----
            USER QUESTION;
            Explain me why a {{profile}} should attend conference {{conference}}.
            The conference is on {{current_date}} at {{current_time}} with {{current_date_time}}
            """);

        Map<String, Object> variables = new HashMap<>();
        variables.put("profile", "Java Developer");
        variables.put("conference", "Devoxx Belgium");

        // Velocity / Mustache ...anyone ^_^ ? oldies but goodies
        Prompt prompt = promptTemplate.apply(variables);
        Response<AiMessage> response = getChatLanguageModel(MODEL_GEMINI_PRO).generate(prompt.toUserMessage());
        prettyPrint(response);
    }

    @Test
    public void system_vs_user_prompts() {
        String systemPrompts = """
            ----
            DIRECTIVE:
            - You are an assistant targeting Java developers
            - Your mission is to provide helpful answers
            ----
            CONSTRAINTS
            - If you do not know, say you do not know
            - Only answers related to java programming languages
            ----
            """;

        PromptTemplate userPromptTemplate = PromptTemplate.from("""
            Explain me why a {{profile}} should attend conference {{conference}}.
            The conference is on {{current_date}} at {{current_time}} with {{current_date_time}}
            """);
        Map<String, Object> variables = new HashMap<>();
        variables.put("profile", "Java Developer");
        variables.put("conference", "Devoxx Belgium");

        // Velocity / Mustache ...anyone ^_^ ? oldies but goodies
        Prompt userPrompt = userPromptTemplate.apply(variables);
        Response<AiMessage> response = getChatLanguageModel(MODEL_GEMINI_PRO).generate(
                SystemMessage.from(systemPrompts),
                userPrompt.toUserMessage());
        prettyPrint(response);
    }

}
