import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;


public class G40HM4 {
    public static void main(String[] args) {

        //------- PARSING CMD LINE ------------
        // Parameters are:
        // <path to file>, k, L and iter

        if (args.length != 4) {
            System.err.println("USAGE: <filepath> k L iter");
            System.exit(1);
        }

        int k = 0, L = 0, iter = 0;
        try {
            k = Integer.parseInt(args[1]);
            L = Integer.parseInt(args[2]);
            iter = Integer.parseInt(args[3]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (k <= 2 && L <= 1 && iter <= 0) {
            System.err.println("Something wrong here...!");
            System.exit(1);
        }
        //------------------------------------

        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //------- SETTING THE SPARK CONTEXT
        SparkConf conf = new SparkConf(true)
                .setAppName("kmedian new approach");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //------- PARSING INPUT FILE ------------
        JavaRDD<Vector> pointset = sc.textFile(args[0], L)
                .map(G40HM4::strToVector)
                .repartition(L)
                .cache();
        long N = pointset.count();
        System.out.println("Number of points is : " + N);
        System.out.println("Number of clusters is : " + k);
        System.out.println("Number of parts is : " + L);
        System.out.println("Number of iterations is : " + iter);

        //------- SOLVING THE PROBLEM ------------
        double obj = MR_kmedian(pointset, k, L, iter);
        System.out.println("Objective function is : <" + obj + ">");
    }

    private static Double MR_kmedian(JavaRDD<Vector> pointset, int k, int L, int iter) {
        long init, fini;


        //------------- ROUND 1 ---------------------------
        init = System.currentTimeMillis();

        JavaRDD<Tuple2<Vector, Long>> coreset = pointset.mapPartitions(x ->
        {
            ArrayList<Vector> points = new ArrayList<>();
            ArrayList<Long> weights = new ArrayList<>();
            while (x.hasNext()) {
                points.add(x.next());
                weights.add(1L);
            }
            ArrayList<Vector> centers = kmeansPP(points, weights, k, iter);
            ArrayList<Long> weight_centers = compute_weights(points, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weight_centers.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        });

        fini = System.currentTimeMillis();
        System.out.println("Round 1 completed in " + (fini - init) + "ms");


        //------------- ROUND 2 ---------------------------
        init = System.currentTimeMillis();

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>(k * L);
        elems.addAll(coreset.collect());
        ArrayList<Vector> coresetPoints = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < elems.size(); ++i) {
            coresetPoints.add(i, elems.get(i)._1);
            weights.add(i, elems.get(i)._2);
        }

        ArrayList<Vector> centers = kmeansPP(coresetPoints, weights, k, iter);

        fini = System.currentTimeMillis();
        System.out.println("Round 2 completed in " + (fini - init) + "ms");


        //------------- ROUND 3: COMPUTE OBJ FUNCTION --------------------
        init = System.currentTimeMillis();

        Double objval = pointset.mapPartitionsToDouble(it ->
                // Convert iterator to stream, map each data point to
                // the value of the distance to its closest center
                // and reduce it taking the sum of all those distances
                StreamSupport.stream(((Iterable<Vector>) () -> it).spliterator(), true)
                        .map(v ->
                                centers.stream()
                                        .map(c -> euclidean(v, c))
                                        .reduce(Double.MAX_VALUE, Double::min)
                        ).iterator()

        ).sum() / pointset.count();

        fini = System.currentTimeMillis();
        System.out.println("Round 3 completed in " + (fini - init) + "ms");

        return objval;
    }

    private static ArrayList<Long> compute_weights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long[] weights = new Long[centers.size()];
        Arrays.fill(weights, 0L);

        for (Vector point : points) {
            double tmp = euclidean(point, centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(point, centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(point, centers.get(j));
                }
            }
            weights[mycenter] += 1L;
        }

        return new ArrayList<>(Arrays.asList(weights));
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];

        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }

        return Vectors.dense(data);
    }

    // Euclidean distance
    private static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    /**
     * Method from G40HM3.java
     * <p>
     * Compute a first set $C'$ of centers using the weighted variant of the kmeans++ algorithm.
     * In each iteration, the probability for a non-center point $p$ of being chosen as next center is
     * $$    w_p \frac{d_p}{\sum_{q \; non-center} w_q d_q}    $$
     * where $d_p$ is the distance of $p$ from the closest among the already selected centers and $w_p$ is the weight
     * of $p$.
     * Compute the final clustering $C$ by refining $C'$ using {@code iter} iterations of Lloyds' algorithm.
     * This implementation of kmeans++ runs in time $O(|P| k)$ because at every iteration we maintain, for each point
     * $q$ in $P$, its distance from the closest center among the current ones.
     *
     * @param P       list of data points to be clustered.
     * @param weights weight of each data point.
     * @param k       number of clusters into which the dataset must be split.
     * @param iter    number of times the Lloyd's algorithm will run to find a good clustering.
     * @return The list of centers.
     */
    private static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> weights, int k, int iter) {
        Random rng = new Random();

        ArrayList<Vector> centers = new ArrayList<>();

        /* We wish to NOT modify ``P`` and ``weights``.
         * For the sake of efficiency, we don't create a deep copy of them,
         * instead we keep track of the indices of the removed candidate centers
         * to push them back later into their original position.
         */
        ArrayList<Integer> cpos = new ArrayList<>();

        // Pick the first center uniformly at random from ``P``
        int rpos = rng.nextInt(P.size());
        final Vector c0 = P.remove(rpos);
        centers.add(c0);
        weights.remove(rpos);
        cpos.add(0, rpos);

        // Compute distances from any point in ``P \ {c0}`` to the first center ``c0``
        ArrayList<Double> distances = new ArrayList<>();
        for (Vector p : P) {
            distances.add(Math.sqrt(Vectors.sqdist(p, c0)));
        }

        // Find all the other ``k - 1`` centers
        for (int i = 1; i < k; i++) {
            /* Compute $sum_{q non center} w_q * d_q$ that will be used to pick
             * random points according to the kmeans++ probability distribution.
             */
            Double distSum = distances.stream().reduce(.0, Double::sum);

            // Pick the next center at random according to the defined distribution
            double r = rng.nextDouble();

            double acc = 0;
            rpos = 0;

            while (rpos < P.size() && r > acc) {
                // Accumulate probabilities $\pi_j$ up to ``rpos``
                acc += distances.get(rpos) * weights.get(rpos) / distSum;
                ++rpos;
            }

            Vector c = P.remove(rpos - 1);
            centers.add(c);
            weights.remove(rpos - 1);
            distances.remove(rpos - 1);
            cpos.add(0, rpos - 1);

            /* Update closest distances from any point in $P \setminus S$ to
             * its closest center in $S$.
             */
            for (int j = 0; j < P.size(); j++) {
                distances.set(j, Math.min(
                        Math.sqrt(Vectors.sqdist(P.get(j), c)),
                        distances.get(j)));
            }
        }

        /* Re-insert the centers and the weights (they were removed in the kmeans++ part).
         *
         * Note: insert in reversed order with respect to the one they were removed,
         *       otherwise they would end up in different (possibly invalid) positions.
         */
        for (int i = 0; i < cpos.size(); ++i) {
            P.add(cpos.get(i), centers.get(i));
            weights.add(cpos.get(i), weights.get(i));
        }

        // Run Lloyd's algorithm up to ``iter`` times, until the objective function does not decrease anymore
        double objVal = Double.POSITIVE_INFINITY;
        for (int it = 0; it < iter; it++) {
            // Label each point in ``P`` to the index of its closest center
            ArrayList<Integer> labels = P.stream()
                    .map(p -> {
                        ArrayList<Double> allDistances = centers.stream()
                                .map(c -> Math.sqrt(Vectors.sqdist(p, c)))
                                .collect(Collectors.toCollection(ArrayList::new));

                        return allDistances.indexOf(Collections.min(allDistances));
                    })
                    .collect(Collectors.toCollection(ArrayList::new));

            // Update each center
            IntStream.range(0, k).forEach(label ->
            {
                Vector newCenter = new DenseVector(new double[P.get(0).size()]);

                /* We calculate the weight of the current cluster, i.e. the one
                 * labeled ``label`` that we later need to calculate the next centroid
                 */
                double clusterWeight = (double) IntStream.range(0, labels.size())
                        .filter(idx -> labels.get(idx) == label)
                        .mapToObj(weights::get)
                        .reduce(0L, Long::sum);

                for (int j = 0; j < P.size(); ++j) {
                    if (labels.get(j) == label) {
                        BLAS.axpy(weights.get(j) / clusterWeight, P.get(j), newCenter);
                    }
                }
                centers.set(label, newCenter);
            });

            // To avoid overhead, we don't update the objective function's value
            if (0 == it % 8) {
                double newObj = kmeansObj(P, centers);

                if (newObj < objVal) objVal = newObj;
                else break;
            }
        }

        return centers;
    }

    /**
     * Method from G40HM3.java
     * <p>
     * Given a clustering, computes the value of the objective function.
     *
     * @param P The points of the dataset.
     * @param C The centers of the clusters.
     * @return The value of the objective function.
     */
    private static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C) {
        double dist = 0;

        for (Vector p : P) {
            dist += C.stream()
                    .map(c -> Math.sqrt(Vectors.sqdist(p, c)))
                    .reduce(Double.MAX_VALUE, Double::min);
        }

        return dist / P.size();
    }
}

