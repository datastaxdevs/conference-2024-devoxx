package devoxx.demo._1_vertexai;

import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import devoxx.demo.utils.AbstractDevoxxTestSupport;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * This test demonstrates how to use a prompt template to generate a prompt for a language model.
 * <a href="https://mustache.github.io/">MUSTACHE</a>
 */
class _16_PromptTemplateTest extends AbstractDevoxxTestSupport {

    @Test
    void prompt() {
        PromptTemplate promptTemplate = PromptTemplate.from("""
            Explain me why a {{profile}} should attend conference {{conference}}.
            The conference is on {{current_date}} at {{current_time}} with {{current_date_time}}
            """);

        Map<String, Object> variables = new HashMap<>();
        variables.put("profile", "Java Developer");
        variables.put("conference", "Devoxx France");

        Prompt prompt = promptTemplate.apply(variables);
        Response<String> response = getLanguageModelTextBison().generate(prompt);
        System.out.println(response.content());
    }
}
