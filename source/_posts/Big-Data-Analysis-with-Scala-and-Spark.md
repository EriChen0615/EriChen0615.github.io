---
title: Big Data Analysis with Scala and Spark
date: 2021-07
tags:
    - Scala
    - Big Data
    - Spark
---

Course Link: https://www.coursera.org/learn/scala-spark-big-data/

# Introduction

## Why Scala? Why Spark?

When the dataset gets too large to fit into memory, languages like R/Python/Matlab will not be able to work with it.

By working in Scala, in a functional style, you can quickly scale your algorithms

In this course you will learn:

1. Extending data parallel paradigm to the distributed case, uisng Spark
2. Spark's programming model
3. Distributing computation, and cluster topology in Spark
4. How to improve performance; data locality, how to avoid recomputation and shuffles in Spark
5. Relational operations with DataFrames and Datasets

Prerequisites:

- Principles of Functional Programming in Scala
- Functional Program Design in Scala
- Parallel Programming (in Scala)

Recommended books: *Learning Spark*, *Spark in Action*, *Advanced Analytics with Spark*

## Data-Parallel to Distributed Data-Parallel

### Shared memory data parallelism

![截屏2021-07-17 下午7.57.09](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%887.57.09.png)

### Distributed data parallelism

Although the code looks identical using the collection abstraction, the internal working is very different. In particular we need to consider the **latency** that is introduced by the network between the nodes.

![截屏2021-07-17 下午8.00.24](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.00.24.png)

### RDD (Resilient Distributed Dataset)

Spark implements a distributed data parallel model called RDD. It is the counterpart of shared memory collection abstraction in data parallelism paradigm.

## Latency

Distribution introduces important concerns beyond what we had to worry about when dealing with parallelism in the shared memory case:

- *Parial failure*: crash failure of a subset of the machines involved in a distributed computation
- *Latency*: certain operations have a much higher latency than other operations due to network communication

Spark handles this two issues particularly well.

> **Latency cannot be masked completely; it will be an important aspect that also impacts the programming model**

![截屏2021-07-17 下午8.14.12](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.14.12.png)

Reading from memory is 100x times faster than reading from disk

If we multiplied the number by **a billion**, we get a humanized latency number

![截屏2021-07-17 下午8.16.03](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.16.03.png)

![截屏2021-07-17 下午8.16.32](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.16.32.png)

![截屏2021-07-17 下午8.17.42](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.17.42.png)

![截屏2021-07-17 下午8.18.30](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.18.30.png)

### Why is Spark faster than Hadoop

**Idea**: Keep all data **immutable and in-memory**. All operations on data are just functional transformations, like regular Scala collections. **Fault tolerance** is achieved by replaying functional transformations over original dataset.

**Result**: Spark has been shown to be 100x more perforant than Hadoop while adding even more expressive APIs

## Basics of Spark's RDDs

RDDs seem like **immutable** sequential or parallel Scala collections.

```scala
abstract class RDD[T] {
	def map[U](f: T => U): RDD[U] = ...
	def flatMap[U](f: T => TraversableOnce[U]): RDD[U] = ...
	def filter(f: T => Boolean): RDD[T] = ...
	def reduce(f: (T, T) => T ): T = ...
}
```

RDD makes heavy use of higher-order functions.

```scala
// word count with RDD

val rdd = sparks.textFile("hdfs://...")
val count = rdd.flatMap(line => line.split(" ")) // separate lines into words
							.map(word => (word, 1)) // include something to count
							.reduceByKey(_ + _) // sum up the 1s in the pairs
```

### Creating RDDs

RDDs can be created in two ways:

1. Transforming an existing RDD

   Just like a call to `map` on a `List`

2. From a `SparkContext` or `SparkSession` object

   The `SparkContext` object (renamed `SparkSession`) can be thought of as your handle to the Spark cluster. It represents the connection between the Spark cluster and your running application. It defines a handful of methods which can be used to create and populate a new RDD:

   - `parallelize`: convert a local Scala collection to an RDD
   - `textFile`: read a text file from HDFS or a local file system and return an RDD of `String`

### Transformations and Actions

**Transformers** return new collections as results (Not single value)

Examples: map, filter, flatMap, groupBy

**Accessors**: Return single values as results (Not collection)

Example: reduce, fold, aggregate

Similarly, Spark defines ***transformations*** and ***actions*** on RDDS

**Transformation**: Return new RDDs as result

> **They are lazy**, their result RDD is not immediately computed

**Actions**: Compute a result based on an RDD,  and either returned or saved to an external storage system

>  **They are eager**, their result is immediately computed

**Laziness / eagerness** is how we can limit network communication using the programming model

```scala
val largeList: List[String] = ...
val wordsRdd = sc.parallelize(largeList)
val lengthsRdd = wordsRdd.map(_.length) // nothing happens in cluster (yet)!
val totalChars = lengthsRdd.reduce(_ + _) // actual computation begins
```

![截屏2021-07-17 下午8.50.18](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.50.18.png)

![截屏2021-07-17 下午8.50.34](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%888.50.34.png)

The action will get the result back to your machine, so typically you will need some transformations beforehand to reduce the size of your RDD.

![截屏2021-07-17 下午9.12.08](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%889.12.08.png)

![截屏2021-07-17 下午9.12.33](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%889.12.33.png)

### Evaluations in Spark: Unlike Scala Collections

Spark is faster in terms of running iterations because it doesn't need to persist the data in HDFS (disk) as in Hadoop but in memory.

![截屏2021-07-17 下午10.16.53](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.16.53.png)

By default, RDDS are recomputed each time you run an action on them. This can be expansive (in time) if you need to a dataset more than once. **Spark allows you to contorl what is cached in memory** using `persist()` or `cache()` 

```scala
val lastYearsLogs: RDD[String] = ...
val logsWithErrors = lastYearsLogs.filter(_.contains("ERROR")).persist()
val firstLogsWithErrors = logsWithErrors.take(10)
val numErrors = logsWithErrors.count() // faster
```

Without the `persist()` we will run the transformation `filter(_.contains("ERROR"))` two times in the above snippet. 

There are many ways to configure how your data is persisted.

- in memory as regular Java objects
- on disk as regular Java objects
- in memory as serialized Java objects (more compact)
- on disk as serialized Java objects (more compact)
- both in memory and on disk (spill over to disk to avoid re-computation)

**cache()**: is the shorthand for using the default storage level, which is in memory only as regular Java objects

**persist**(): Persistance can be customized with this method. Pass the storage level you'd like as a parameter to `persist`.

![截屏2021-07-17 下午10.29.50](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.29.50.png)

The default is the **MEMORY_ONLY** storage level.

> The deferred semantics of Spark's RDDs are very unlike Scala Collections

**...One of the most common performance bottlenecks of newcomers of Spark arises from unknowingly re-evaluating several transformations when caching could be used.**

Lazy evaluation allows Spark to **stage** computations and make important **optimization** to the **chain of operations** before execution (e.g., `map()` followed by `filter() `can be traversed once only)

## Cluster Topology Matters!

An example to kick start:

```scala
case class Person(name: String, age: Int)
val people: RDD[Person] = ...
people.foreach(println) // this prints in the executor, can't see in the driver (master)
val first10 = people.take(10) // this ends up in the driver program 
```

The atonomy of a Spark job.

![截屏2021-07-17 下午10.42.55](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.42.55.png)

![截屏2021-07-17 下午10.43.05](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.43.05.png)

![截屏2021-07-17 下午10.43.59](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.43.59.png)

![截屏2021-07-17 下午10.44.28](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.44.28.png)

![截屏2021-07-17 下午10.44.52](%E6%88%AA%E5%B1%8F2021-07-17%20%E4%B8%8B%E5%8D%8810.44.52.png)

# Reduction Ops & Distributed Key-Value Pairs

## Reduction Operations

Operations such as `fold`, `reduce` and `aggreate` from Scala sequential collections.

They **walk through a collection and combine neighbouring. elements of the collection together to product a single combined result**

```scala
case class Taco(kind: String, price: Double)
val tacoOrder = 
  List(
  Taco("...", 2.25),
  Taco("xxx", 1.1)
 )

val cost = tacoOrder.foldLeft(0)((sum, taco) => sum + taco.price)

```

`foldLeft` is not parallelizable due to its signature `def foldLeft[B](z:B)(f: (B,A) => B ): B`, because if you try to parallelize the computation, the types wouldn't match in the combine stage.

![截屏2021-07-18 下午3.36.51](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%883.36.51.png)

`fold` enables us to parallelize things. Because it's signiture restricts us to always returning the same type `def fold(z: A)(f: (A,A) => A): A`

`aggregate` lets you still do sequential-style folds in chunks and then combine them together. It is the generalization of `foldLeft`

`aggregate[B] (z: => B) (seqop: (B, A) => A, combop: (B,B) => B): B`

Spark doesn't give you the option to use `foldLeft/foldRight`, which means that if you have to change the return type of your reduction operation, your only choice is `aggregate`

> It simply doesn't make sense to enforce sequential execution across a cluster.

## Distributed Key-Value Pairs (Pair RDDs)

In single-node Scala, key-value pairs can be thought of as **maps**

Large datasets are often made up of unfathomably large numbers of complex, nested data records. To be able to work with such datasets, it's often desirable to project down these complex datatypes into **key-vlaue pairs**

![截屏2021-07-18 下午3.47.18](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%883.47.18.png)

Pair RDDs have additional, specialized methods for working with data associated with keys `RDD[(K,V)]`. Commonly used ones are `groupBykey(), reduceByKey, join()`

### Creating a Pair RDD

Pair RDDs are most often created from already-existing non-pair RDDs, for example by using the `map` operation on RDDs.

```scala
val rdd: RDD[WikipediaPage]

val pairRdd = rdd.map(page => (page.title, page.text))
// now you have a lot more operations at hand!
```

### Transformation and Action for pair RDD

Transformations:

- groupByKey
- reduceByKey
- mapValues
- keys
- join
- leftOuterJoin/rightOuterJoin

#### **groupByKey**

*Regular Scala*

`def groupBy[K](f: A => K): Map[K, Traversable[A]]`

Partions this traversable collection into a map of traversable collections according to some discriminator function.

```scala
val ages = List(2, 52, 44, 23, 88)
val grouped = ages.groupBy{ age =>
	if (age >= 18 && age < 65) "adult"
	else if (age < 18) "child"
	else "senior"
}
// Map(senor->List(88), audlt -> List(52, 44, 23), child -> List(2))
```

*Spark*

`def groupByKey(): RDD[(K, Iterable[V])]`

It is specialized at collection values of the same key, because the RDD is already paired, we don't need a discrimitive function.

```scala
case class Event(organizer: String, name: String, budget: Int)
val eventsRdd = sc.parallelize(...)
								.map(event => event.organizer, event.budget)

val groupedRdd = eventsRdd.groupByKey() // LAZY!!!
groupedRdd.collect().foreach(println) // executed here
```

#### **reduceByKey**

Conceptually, `reduceByKey` can be thought of as a combination of `groupByKey` aand `reduce`-ing on all the values per key.

> But `reduceByKey` is much more efficient than applying separate steps!!!

`def reduceByKey(func: (V,V) => V): RDD[(K,V)]`

```scala
case class Event(organizer: String, name: String, budget: Int)
val eventsRdd = sc.parallelize(...)
								.map(event => event.organizer, event.budget)

val budgetsRdd = eventsRdd.reduceByKey(_ + _)
budgetsRdd.collect().foreach(println)
```

#### **mapValues**

`def mapValues[U] (f: V => U): RDD[(K,U)]` can be thought of as a shorthand for `rdd.map{ case(x,y): (x, func(y))}` That is, it simply applies a function to only the values in a Pair RDD.

#### **countByKey**

`def countByKey(): Map[K, Long]` simply counts the number of elements per key in a pair RDD, returning a normal scala map.

Let's compute the **average budget of the organziers**

```scala
val res = 
	eventsRdd.mapValues(b => (b,1))
  .reduceByKey((v1, v2) => (v1._1+v2._1, v1._2 + v2._2))
  .mapValues(case (total, cnt) => total/cnt )
	.collect().foreach(println)
```

#### **keys** 

`def keys; RDD[K]` returns an RDD with the keys of each tuple

> This method is a transformation and thus returns an RDD because the number of keys in a pair RDD may be unbounded.

```scala
case class Visitor(ip: String, timestamp: String, duration: String)
val visits: RDD[Visitor] = sc.textfile(...)
val numUniqueVisits = visits.keys.distinct().count()
```

#### Joins

Joins are another sort of transformation on pair RDDS. They're used to combine multiple datasets . They are one of the most commonly-used operations on Pair RDDs

There are two kinds of joins:

- Inner joins `join`
- Outer joins `leftOuterJoin / rightOuterJoin`

The key difference between the two is what happens to the keys when both RDDs don't contain the key

![截屏2021-07-18 下午4.23.51](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%884.23.51.png)

**Inner Joins**

Inner joins returns a new RDD containing combined pairs whose **keys are present in both input RDDs**

`def join[W] (other: RDD[(K,W)]): RDD[(K, (V, W))]`

```scala
val abos = ...
val locations = ...

val trackedCustomers = 
	abos.join(location)
	.keys.distinct().count()										
```

> Inner joins are **lossy**

**Outer Joins (leftOuterJoin, rightOuterJoin)**

Outer joins return a new RDD containing combined pairs **whose keys don't have to be present in both input RDDS**

`def leftOuterJoin[W] (other: RDD[(K,W)]): RDD[(K, (V, Option[W]))]`

`def rightOuterJoin[W] (other: RDD[(K,W)]): RDD[(K, (Option[V], W))]`

![截屏2021-07-18 下午4.33.45](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%884.33.45.png)



# Partition and Shuffling

## Shuffling 

Shuffles happen with operation like `groupByKey` because data has to move around the network. 

![截屏2021-07-18 下午4.42.27](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%884.42.27.png)

Shuffling is bad because the network transport is very slow (in humanized scale from seconds to days)

The `reduceByKey` operation will **reduce whenever possible to reduce the amount of data shuffled between the nodes**.

![截屏2021-07-18 下午6.07.43](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.07.43.png)

But how does Spark know which key to put on which machine?

## Partitioning

The dat withint an RDD is split into many *partitions*

- Partitions never span multiple machines. 
- Each machine in the cluster contains one or more partitions
- THe number of partitions to use is configurable. By default, it equals the *total number of cores on executor nodes*

There are two kinds of partitioning available in Spark:

- Hash partitioning
- Range partitioning

> Partitioning only works with pair RDDs

### Hash Partitioning

Hash partitioning attempts to spread data evenly across partitions *based on the keys*. 

### Range Partitioning

Pair RDDs may contain keys that have an ordering defined: Int, Char, String, ...

For such RDDs, *range partitioning* may be more efficient -> tuples with keys in the same range will live on the same node

### Partitioning Data

1. Call `partitionBy`, providing specific `partitioner`
2. Using transformations that return RDDs with specific partitioners

```scala
val pairs = purchasesRdd.map(p => (p.customerId, p.price))

val tunedPartitioner = new RangePartitioner(8, pairs)
val partitioned =
	pairs.partitionBy(tunedPartitioner)
	.persist() // persist! Otherwise shuffled every time
```

> Important: the result of the partition should always be persisted

Partitioner from parent RDD:



Automatically-set partitioners:

`sortByKey` invokes a `RangePartitioner`

![截屏2021-07-18 下午6.22.01](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.22.01.png)

**All other operations will produce a result without a partitioner**

Make sure you use these transformations if you want to keep the partition. Notice `map` and `flatMap` are **NOT** on the list. Because `map` and `flatMap` can change the keys of the pair RDDs.

```scala
rdd.map((k: String, v: Int) => ("doh!", v))
```

That is why you should always try to use `mapValues` if possible, because it's impossible to change the key, hence keeping the partitioning.

### Improve Efficiency with Partitioning

From *Learning Spark* Page 61- 64

![截屏2021-07-18 下午6.34.32](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.34.32.png)

There are two common scenarios where partition can reduce shuffling:

1. `reduceByKey` running on a pre-partitioned RDD will cuase the values to be computed **locally**.
2. `join` called on two RDDs that are pre-partitioned with the same partitioner are cached on the same machine will cause the join to be computed **locally**.

## Know when shuffle will occur 

**A shuffle *can* occur when the resulting RDD depends on other elements from the same RDD or another RDD.** Paritioning is often the solution.

Or look at the return type, or use the function `toDebugString` to see its execution plan.

![截屏2021-07-18 下午6.36.33](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.36.33.png)

![截屏2021-07-18 下午6.38.17](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.38.17.png)

# Wide vs Narrow Dependencies

Some transformationnns are significantly more expensive than others 

## Linages 

Computations on RDDs are represented as a **lineage graph**; a Direct Acyclic Graph (DAG) representing the computations done on the RDD.

![截屏2021-07-18 下午6.43.27](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.43.27.png)

## How are RDDs represented?

RDDs are made up of 4 important parts

- **Partitions**: Atomic pieces of the dataset. One or many per compute node
- **Dependencies**: Models relationship between this RDD *and its partitions* with the RDD(s) it was derived from
- **A function**: for computing the dataset based on its parent RDDs
- **Metadata** about its partitioning scheme

![截屏2021-07-18 下午6.46.45](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.46.45.png)

![截屏2021-07-18 下午6.47.45](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.47.45.png)

RDD dependencies encode when data must move across the network. There are two kinds of dependencies:

1. **Narrow Dependency**: Each partition of the parent RDD is used by at most one partition of the child RDD 
2. **Wide Dependency**: Each partition of the parent RDD may be depended on by **multiple** child partitions

![截屏2021-07-18 下午6.49.58](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.49.58.png)



![截屏2021-07-18 下午6.57.07](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.57.07.png)

![截屏2021-07-18 下午6.59.30](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%886.59.30.png)

![截屏2021-07-18 下午7.00.03](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%887.00.03.png)

You can use `dependencies` method on RDDs to figure out the dependency

```scala
val pairs = wordsRdd.map(c => (c,1))
										.groupByKey()
										.dependencies // or toDebugString
```

The *lineage graph* is composed of *stages*. 

 ## Fault Tolerance

![截屏2021-07-18 下午7.06.09](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%887.06.09.png)

![截屏2021-07-18 下午7.07.07](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%887.07.07.png)

Recomputing missing partitions is fast for narrow dependencies, but slow for wide dependencies.

# SQL, Dataframes and Datasets 

Given a bit of extra structural information, Spark can do many optimizations for you.

**Unstructured**: Log files, images

**Semi-structured**: json, xml (self-describing)

**Structured**: database tables 

Spark + regular RDDs don't know anything about the **schema** of the data it's dealing with.

![截屏2021-07-18 下午8.21.33](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.21.33.png)

**Structured vs Unstructured Computation**

In Spark: we do functional transformations on data. We pass user-defined function literals to higher-order functions like `map`, `flatMap` and `filter`

In database/Hive: We do declarative transformation on data.

![截屏2021-07-18 下午8.24.15](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.24.15.png)

## Spark SQL

SQL is the lingua franca for doing analytics. But it's a pain in the neck to connect big data processing pipelines like Spark or Hadoop to an SQL database

Spark SQL makes it possible to seamlessly **intermix** SQL queries with Scala and to get all of the **optimization** we're used to in the databases community on Spark Jobs. 

Three main goals:

1. Support **relational processing** both within Spark programs (on RDDs) and on external data sources with a friendly API
2. High performance
3. Easily support new data sources such as semi-structured data and external databases

Three main APIs:

1. SQL literal syntax
2. DataFrames
3. Datasets

Two specialized backend components:

1. Catalyst, query optimizer
2. Tungsten, off-heap serializer

![截屏2021-07-18 下午8.30.39](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.30.39.png)

To get started using Spark SQL, everything starts with `SparkSession`. (rather than `SparkContext`)

## DataFrame

**DataFrame** is Spark SQL's core abstraction. DataFrames are conceptually RDDs full of records with a known schema. DataFrames are **untyped**

Transformations on DataFrames are also known as **untyped transformation**

### Creating DataFrames

1. From an existing RDD `toDF()`, or explicitly specify the schema
2. Reading in specific **data source** from file

toDF()

```scala
val tupleRDD = ...
val tupleDF = tupleRDD.toDF("id", "name", "city", "country")
```

```scala
val schemaString = "name age"
val fields = schemaString.split(" ")
 .map(fieldName => StructField(fieldName, StringType, nullable = true))

val schema = StructType(fields)

val rowRDD = peopleRDD
	.map(_.split(","))
	.map(attributes => Row(attributes(0), attributes(1).trim))
	
val peopleDF = spark.createDataFrame(rowRDD, schema)
```

Semi-structured sources: Json, csv, parquet, JDBC

### SQL Literals

```scala
peopleDF.createOrReplaceTempView("people")
val adultsDF
 = spark.sql("SELECT * FROM people WHERE age > 17")
```

The SQL statements available to you are largely what's available in HiveQL. 

## DataFrames API

DataFrames are **a relational API over Spark's RDDs**

### DataFrames Data Types

![截屏2021-07-18 下午8.47.23](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.47.23.png)

Complex Spark SQL Data Types

![截屏2021-07-18 下午8.49.55](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.49.55.png)

![截屏2021-07-18 下午8.51.54](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.51.54.png)

It is possible to arbitrarily nest complex data types

> In order to use Spark SQL Types you need to import `import org.apache.spark.types._`

### Common Transformations

Common transformation includes `select` ,`agg`, `groupBy` and `join`

![截屏2021-07-18 下午8.56.51](%E6%88%AA%E5%B1%8F2021-07-18%20%E4%B8%8B%E5%8D%888.56.51.png)

`select` selects a set of named columns and returns a new DataFrame with these columns as a result

`agg` performs aggregations of a series of columns and returns a new DataFrame with the calculated output

`groupBy` groups the DataFrame using the specified columns. **Intended to be used before an aggregation**

`join` inner join with another DataFrame

Other transformations include: `filter, limit, orderBy, where, as, sort, union, drop` etc 

> **Three ways to specify Columns:** 
>
> 1. Using \$-notation: `df.filter($"age" > 18)` It requries `import spark.implicits._`
> 2. Referring to the DataFrame `df.filter(df("age" > 18))`
> 3. Using SQL query string `df.filter("age> 18")`

An example:

```scala
case class Employee(id: Int, fname: String, lname: String, age: Int, Stirng)
val sydneyEmployeesDF = employeeDF.select("id", "lanme")
																	.where("city == 'sydney'")
																	.orderBy("id")
```

`filter` and `where` are the same

> Note that to use equal condition for filter, the syntax is
>
> ```scala
> df.filter($"col_name" === 'value') // note '==='!
> ```

### Grouping and Aggregating on DataFrames

One of the most common tasks on tables is to (1) group data by a ceratin attribute, and then (2) do some kind of aggregation on it like a count

a `groupy` function which returns a `RelationalGroupedDataset` 

**Example**

Compute the most expensive and least expensive homes for sale per zip code.

```scala
case class Listing(street: String, zip: Int, price: Int)
val listingsDF = ... // DataFrame of Listing

import org.apache.spark.sql.functions._
val mostExpensiveDF = listingsDF.groupBy($"zip")
																.max("price")
val mostExpensiveDF = listingsDF.groupBy($"zip")
																.min("price")				
```

```scala
case class Post(authorID: Int, subforum: String, likes: Int, date: String)
val postDF = ... // DataFrame of Posts

import org.apache.spark.sql.functions._

val rankedDF = 
 postsDF.groupBy($"authorID",$"subforum")
 				.agg(count($"authorID")) // new DF with columns authorID, subforum, count(authorID)
 				.orderBy($"subforum", $"count(authorID)".desc)
```

### Getting a look at your data

You can us `show()` which pretty prints the DataFrame in tabular form (by default 20 row). `printSchema()` prints the schema of your DataFrame in a tree format.

## Closer look in DataFrame

### Cleaning Data with DataFrames

It's desirable to do one of the following:

- drop row/records with unwanted values like `null` or `NaN`
- replace certain values with a constant

**Dropping**

We can use `drop()` which drops rows that contain `null` or NaN values in **any** columns and returns a new DataFrame

`drop("all")` drops rows that contain `null` or `NaN` values in **all** columns

`drop(Array("id", "name"))` drops rows that contain `null` or `NaN`values in the **specified** columns and returns a new `DataFrame`

**Replacing**

`fill(0)` replaces all occurrences of `null` or `NaN` in **numeric columns** with **specified vlaue** and returns a new `DataFrame`

`fill(Map("column_name"->0))` replaces all occurrences of `null` or `NaN` in **specified column** with **specified value** and returns a new `DataFrame`

`replace(Array("id"), Map(1234->8923))` replaces **specified value** in **specified column** with **specified replacemet value** and returns a new `DataFrame`

### Common Actions on DataFrames

`collect(): Array[Row]` Returns an array that contains all of Rows in this DataFrame

`count(): Long` 

`first(): Row/head(): Row`: returns the first row in the DataFrame

`show(): Unit`: Displays the top 20 rows of DataFrame 

`take(n: Int): Array[Row]`

### Joins on DataFrames

We need to specify which columns we should join on. `inner`, `outer`, `left_outer`, `right_outer`, `left_semi`

```scala
df1.join(df2, $"df1.id" === $"df2.id")
df1.join(df2, $"df1.id" === $"df2.id", "right_outer")
```

> Different methods are specified as an argument

### Optimizations

**Catalyst**: query optimizer

*Reordering operations* (Laziness + structure gives us the ability to analyze and rearrange DAG of computation, often pushing filter up)

Reduce the amount of data we must read

**Tungsten**: off-heap data encoder (searializer)

- highly-specialized data encoders
- **column-based**
- off-heap (free from garbage collection overhead)

> Tungsten can take schema information and tightly pack serialized data into memory. This means more data can fit in memory and faster serialization/deserialization.
>
> Column-based data storage is well-known to be more efficient across DBMS

### Limitations of DataFrmaes

DFs are **untyped**.

```scala
listingDF.filter($"state" === "CA") // compile, but state can be non-existent
```

We can only use a **Limited Data Types**. This is can be hard when you already uses some kind of complicated regular Scala class

DF requires **Semi-structured / Structured Data**. There might be no structure to it. 

## Datasets

```scala
val averagePrices = averagePricesDF.collect()
// averagePrices: Array[org.apache.spark.sql.Row]

val averagePrices = averagePrice.map {
  row => (row(0).asInstanceOf[Int], row(1).asInstanceOf[Double])
} // inconvenient
```

We can check the schema using

```scala
averagePrices.head.printSchema()
```

The motivation is that we want type safety as well as the optimization.

DataFrame is **untyped**. Dataset is **typed**.  DataFrame is actually a dataset

```
DataFrame = Dataset[Row]
```

A Dataset is a **typed** distributed collections of data. It unifies the `DataFrame` and `RDD` APIs. It requires structured/ semi-structured data. Schemas and `Encoder` s are core parts of `Dataset` 

Datasets are something in the middle between DataFrames and RDDs. You can use relational operations and functional operations like `map`, `flatMap` and `filter`. It's a good choice when you need to mix and match.

### Creating Dataset

**From a Dataset**

```scala
myDF.toDS	 // requires import spark.implicits._
```

**From File**

```scala
val myDS = spark.read.json("people.json").as[Person]
// if case class Person match 
```

**From an RDD & Common Scala types**

```scala
.toDS
```

### Typed Columns

A `TypedColumn` is different from `Column` , you can cast it with 

```scala
$"price".as[Double] // this is a TypedColumn
```

### Transformations on Dataset

Dataset introduces **typed transformations**. 

```scala
map[U](f: T=>U): Dataset[U]
flatMap[U](f: T=>TraversableOnce[U]): Dataset[U]

groupByKey[K](f: T => K): KeyValueGroupedDataset[K,T]
```

### Grouping Operations on Datasets

Calling `groupByKey` on a `Dataset` returns a `KeyValueGroupedDataset`, it contains a number of aggregation operations which return `Datasets`



# Practicals

## Create Column with literal values

```scala
df2 = df1.withColumn("col_name", lit("value"))
df2 = df1.select($"*", lit("value").as("col_name"))
```



