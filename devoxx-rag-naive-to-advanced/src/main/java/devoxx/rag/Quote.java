package devoxx.rag;

import java.util.List;

/**
 * Philisopher quote from a JSON file
 *
 * @param rowId
 *      identifier for the record
 * @param author
 *      author
 * @param tags
 *      tag list
 * @param body
 *      text to be indexed
 */
public record Quote(String rowId, String author, List<String> tags, String body) {}
