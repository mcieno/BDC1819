Big Data Computing
==================

Homeworks for Big Data Computing course (A.Y. 2018/19, _University of Padova_).

-------------

## Homework 1: Introduction to Spark

The purpose of this first homework is to set up the environment for developing Spark code on your machine and to get acquainted with the principles of functional programming, on which MapReduce and Spark are based.


## Homework 2: MapReduce with Spark
In this second homework, we will look at Spark in more details and will learn how to implement MapReduce computations in Spark.


## Homework 3: efficient sequential solution to k-median
The most accurate approximation algorithms for k-median known to date tend to be very slow and inadequate for a big data scenario. The purpose of the homework is to implement sequentially an efficient solution for k-median based on the k-means++ strategy. The implementation will turn out useful for the last homework. For reviewing the theoretical grounds, refer to the slides on Clustering - Part 2.


## Homework 4: k-Median Clustering on a Cloud
This homework will show how the coreset-based approach enables an effective and efficient solution to the k-median clustering problem. In particular, you will test the MapReduce algorithm MR-kmedian described in class (Slides 45-46 of the set Clustering - Part 2) using the implementation of kmeans++ devised for Homework 3 as a primitive (called B in the slides).

Motivation. A recent work presented at the prestigious ACM KDD'17 Symposium (H.Song et al. PAMAE: Parallel k-Medoids Clustering with High Accuracy and Efficiency) compared different distributed approaches for this task, but I suspect that the one you will test may beat them!
