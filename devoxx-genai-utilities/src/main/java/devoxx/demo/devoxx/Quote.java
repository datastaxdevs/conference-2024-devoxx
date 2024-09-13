package devoxx.demo.devoxx;

import java.util.List;

public record Quote(String rowId, String author, List<String> tags, String body) {}
