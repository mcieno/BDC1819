import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
// import org.apache.log4j.Level;
// import org.apache.log4j.Logger;
import scala.Tuple2;

import java.util.*;


public class G40HM2 {
    public static void main(String[] args) {
        // We'll use this pair of variables as a stopwatch
        long init, fini;

        /*
         * Uncomment the following two lines to disable verbose logging
         */
        // Logger.getLogger("org").setLevel(Level.OFF);
        // Logger.getLogger("akka").setLevel(Level.OFF);


        // Parse command line parameters
        if (args.length != 2) {
            throw new IllegalArgumentException(
                    "Expecting filename (args[0]:String) and parameter K (args[1]:int) on the command line");
        }

        String filename = args[0];
        int K = Integer.parseInt(args[1]);

        // Configure Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("Homework 2 - Group 40")
                .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Read the document into a JavaRDD
        JavaRDD<String> docs = sc.textFile(filename).repartition(K);

        /*
         * IMPROVED WORD-COUNT 1
         * =====================
         *
         */
        JavaPairRDD<String, Long> wordCount1 = docs
                /*
                 * ROUND 1 - MAP PHASE
                 * Map all documents into K-V pairs so that the key is the word and the
                 * value is the number of occurrences of such word in the current document.
                 *
                 * To fasten this process, which repeatedly requires a search through all
                 * the already-inserted K-V pairs, we use an HashMap and take advantage
                 * of the method ``getOrDefault``.
                 */
                .flatMapToPair(document -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })

                /*
                 * ROUND 1 - REDUCE PHASE
                 * K-V pairs are grouped by key (i.e. the words)
                 * so we can simply take the sum of each iterator.
                 */
                .reduceByKey(Long::sum);

        init = System.currentTimeMillis();
        long count1 = wordCount1.count();
        fini = System.currentTimeMillis();

        // Calculate required time
        long t1 = fini - init;

        // Calculate average word length
        double avgLength1 = wordCount1.map(p -> (long) p._1.length()).reduce(Long::sum) / (double) count1;

        /*
         * IMPROVED WORD-COUNT 2
         * =====================
         *
         */
        Random prng = new Random();
        JavaPairRDD<String, Long> wordCount2 = docs
                /*
                 * ROUND 1 - MAP PHASE
                 * Map all documents into K-V pairs so that the key is a uniform random
                 * number in $$[0, K)$$ and the value is the pair ``word, count(word)``.
                 *
                 * Again, to fasten this process, which repeatedly requires a search through
                 * all the already-inserted K-V pairs, we use an HashMap and take advantage
                 * of the method ``getOrDefault``.
                 */
                .flatMapToPair(document -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        /*
                         * The new pair is formed by an integer key, randomly generated
                         * and, as value, by a pair that contain a word and a counter of
                         * its occurrences.
                         */
                        pairs.add(new Tuple2<>(prng.nextInt(K), new Tuple2<>(e.getKey(), e.getValue())));
                    }
                    return pairs.iterator();
                })

                /*
                 * ROUND 1 - REDUCE PHASE
                 * Now the RDD is grouped by the random key and we must
                 * generate the new K-V pairs where K is the word and V
                 * is the number of occurrences of that word.
                 *
                 * We exploit again the properties and the efficiency of
                 * a Java HashMap.
                 */
                .groupByKey()
                .flatMapToPair(pairs -> {
                    ArrayList<Tuple2<String, Long>> reversed = new ArrayList<>();
                    HashMap<String, Long> counts = new HashMap<>();

                    for (Tuple2<String, Long> pair : pairs._2) {
                        counts.put(pair._1, pair._2 + counts.getOrDefault(pair._1, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        reversed.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return reversed.iterator();
                })

                /*
                 * ROUND 2 - MAP PHASE
                 * The map function used in second round is the identity
                 * function. Hence no operation is explicitly performed.
                 *
                 * ROUND 2 - REDUCE PHASE
                 * K-V pairs are grouped by key (i.e. the words)
                 * so we can simply take the sum of each iterator.
                 */
                .reduceByKey(Long::sum);

        init = System.currentTimeMillis();
        long count2 = wordCount2.count();
        fini = System.currentTimeMillis();

        // Calculate required time
        long t2 = fini - init;

        // Calculate average word length
        double avgLength2 = wordCount2.map(p -> (long) p._1.length()).reduce(Long::sum) / (double) count2;

        /*
         * IMPROVED WORD-COUNT 2 bis
         * =========================
         *
         */
        JavaPairRDD<String, Long> wordCount3 = docs
                /*
                 * ROUND 1 - MAP PHASE
                 * Map all documents into K-V pairs so that the key is the word and the
                 * value is the number of occurrences of such word in the current document.
                 *
                 * Again, to fasten this process, which repeatedly requires a search through
                 * all the already-inserted K-V pairs, we use an HashMap and take advantage
                 * of the method ``getOrDefault``.
                 */
                .flatMapToPair(document -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })

                /*
                 * ROUND 1 - REDUCE PHASE
                 * The reduce phase is performed exploiting the partitioning already
                 * done by Spark: the operations are similar to the ones in the previous
                 * part, but here the random key is not used since Spark automatically
                 * simulates this mechanism efficiently.
                 */
                .mapPartitionsToPair(it -> {
                    HashMap<String, Long> counts = new HashMap<>();

                    // Iterate all elements in the partition and update ``counts`` accordingly
                    it.forEachRemaining(pair ->
                            counts.put(pair._1, pair._2 + counts.getOrDefault(pair._1, 0L))
                    );

                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })

                /*
                 * ROUND 2 - MAP PHASE
                 * The map function used in second round is the identity
                 * function. Hence no operation is explicitly performed.
                 *
                 * ROUND 2 - REDUCE PHASE
                 * K-V pairs are grouped by key (i.e. the words)
                 * so we can simply take the sum of each iterator.
                 */
                .reduceByKey(Long::sum);

        init = System.currentTimeMillis();
        long count3 = wordCount3.count();
        fini = System.currentTimeMillis();

        // Calculate required time
        long t3 = fini - init;

        // Calculate average word length
        double avgLength3 = wordCount3.map(p -> (long) p._1.length()).reduce(Long::sum) / (double) count3;

        // Pretty print results
        System.out.printf(
                "+---------------------------+---------------------+--------------------+\n" +
                "|         Algorithm         | Average word length | Time required [ms] |\n" +
                "+===========================+=====================+====================+\n" +
                "| Improved word-count 1     | %19.4f | %18d |\n" +
                "| Improved word-count 2     | %19.4f | %18d |\n" +
                "| Improved word-count 2-bis | %19.4f | %18d |\n" +
                "+---------------------------+---------------------+--------------------+\n",
                avgLength1, t1,
                avgLength2, t2,
                avgLength3, t3
        );
    }
}
