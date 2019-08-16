import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class G40HM3 {
    public static void main(String[] args) throws IOException {
        // Read the input passed on the command line
        if (args.length != 3) {
            throw new IllegalArgumentException(
                    "Expecting FILENAME (args[0]:String), NUMBER OF CLUSTERS (args[1]:int) " +
                            "and MAXIMUM ITERATIONS (args[2]:int) on the command line");
        }
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int iter = Integer.parseInt(args[2]);

        // Import the dataset, points and weights. Note, the weights are all equal to 1
        ArrayList<Vector> points = readVectorsSeq(filename);
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            weights.add(1L);
        }

        // Run and time the algorithm to find the centers
        long init = System.currentTimeMillis();
        ArrayList<Vector> centers = kmeansPP(points, weights, k, iter);
        long fini = System.currentTimeMillis();
        System.out.println("Clustering computed in " + (fini - init) + "ms");

        // Compute the objective function value
        System.out.println("Results of kmeans with k=" + k + " and a limit of " +
                iter + " iterations: " + kmeansObj(points, centers));
    }

    /**
     * Convert a string composed by doubles into a Vector. Doubles are separated by a single space.
     *
     * @param str The string that contains the sequence of doubles.
     * @return The corresponding vector.
     */
    private static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];

        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    /**
     * Convert a file into an ArrayList of Vectors. Each line of the file is converted into a Vector,
     * using the strToVector function.
     *
     * @param filename The name of the file to convert.
     * @return The Vectors contained in the file as strings.
     * @throws IOException If the filename passed does not indicate a file (e. g. a directory).
     */
    private static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }

        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(G40HM3::strToVector)
                .forEach(result::add);
        return result;
    }

    /**
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
