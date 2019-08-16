import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;


public class G40HM1 {
    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        // Read a list of numbers from the program options
        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s = new Scanner(new File(args[0]));
        while (s.hasNext()) {
            lNumbers.add(Double.parseDouble(s.next()));
        }
        s.close();

        // G40HW1Setup Spark
        SparkConf conf = new SparkConf(true).setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);

        // Find the max value of the RDD using two different methods
        double maxValueRed = dNumbers.reduce(Math::max);
        double maxValueCmp = dNumbers.max((Serializable & Comparator<Double>) Double::compare);

        System.out.println("Max value obtained with \"reduce\" method " + maxValueRed);
        System.out.println("Max value obtained with \"max\" method " + maxValueCmp);

        // Create normalized RDD
        JavaRDD<Double> dNormalized = dNumbers.map((x) -> x / maxValueCmp);

        double dHarmonicMean = dNormalized.count() / dNormalized
                .map((x) -> 1 / x)
                .reduce((x, y) -> x + y);
        System.out.println("Harmonic Mean: " + dHarmonicMean);
    }
}
