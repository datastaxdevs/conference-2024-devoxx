package devoxx.rag.experiments;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Utils {
    public static void main(String[] args) {

        List<String> segments = List.of("abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz");

        slidingWindow(segments, 1, 2).forEach(System.out::println);

    }

    public static <T> List<List<T>> slidingWindow(List<T> input, int nbBefore, int nbAfter) {
        return IntStream.range(0, input.size())
            .mapToObj(i -> IntStream.rangeClosed(i - nbBefore, i + nbAfter)
                .filter(j -> j >= 0 && j < input.size())
                .mapToObj(input::get)
                .collect(Collectors.toList()))
            .collect(Collectors.toList());
    }
}