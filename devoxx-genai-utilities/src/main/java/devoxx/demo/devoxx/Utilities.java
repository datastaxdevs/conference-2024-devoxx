package devoxx.demo.devoxx;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class Utilities {

    public static final String GCP_PROJECT_ID        = "devoxxfrance";
    public static final String GCP_PROJECT_PUBLISHER = "google";
    public static final String GCP_PROJECT_ENDPOINT  = "us-central1-aiplatform.googleapis.com:443";
    public static final String GCP_PROJECT_LOCATION  = "us-central1";

    public static final Integer EMBEDDING_DIMENSION = 768;
    public static final String  TABLE_NAME          = "vector_store";

    public static final String ASTRA_TOKEN           = System.getenv("ASTRA_DB_APPLICATION_TOKEN");
    public static final String ASTRA_DB_ID           = "bace77c5-80ea-4bc4-a0f4-529121918cd4";
    public static final String ASTRA_DB_REGION       = "us-east1";
    public static final String ASTRA_KEYSPACE        = "default_keyspace";
    public static final String ASTRA_API_ENDPOINT    = "https://"+ ASTRA_DB_ID +"-"+ ASTRA_DB_REGION +".apps.astra.datastax.com/api/json";

    private Utilities() {}

    @SuppressWarnings("unchecked")
    public static  List<Quote> loadQuotes(String filePath) throws IOException {
        File inputFile = new File(Utilities.class.getClassLoader().getResource(filePath).getFile());
        LinkedHashMap<String, Object> sampleQuotes = new ObjectMapper().readValue(inputFile, LinkedHashMap.class);
        List<Quote> result  = new ArrayList<>();
        AtomicInteger quote_idx = new AtomicInteger(0);
        ((LinkedHashMap<?,?>) sampleQuotes.get("quotes")).forEach((k,v) -> {
            ((ArrayList<?>)v).forEach(q -> {
                Map<String, Object> entry = (Map<String,Object>) q;
                String author = (String) k;//(String) entry.get("author");
                String body = (String) entry.get("body");
                List<String> tags = (List<String>) entry.get("tags");
                String rowId = "q_" + author + "_" + quote_idx.getAndIncrement();
                result.add(new Quote(rowId, author, tags, body));
            });
        });
        return result;
    }




}
