---
title: complex datatype in dataframe
date: 2021-07
tags:
    - Big Data
    - Scala
    - Spark
---

# Data Sources and formats

![截屏2021-07-20 上午10.02.11](%E6%88%AA%E5%B1%8F2021-07-20%20%E4%B8%8A%E5%8D%8810.02.11.png)





# Interchanging data formats with Spark SQL

Spark can be used to interchange data formats as easily as:

```scala
events = spark.readStream \
  .format("json") \           # or parquet, kafka, orc...
  .option() \                 # format specific options
  .schema(my_schema) \        # required
  .load("path/to/data")

output = …                   # perform your transformations

output.writeStream \          # write out your data 
  .format("parquet") \
  .start("path/to/write")
```

# Transforming Complex Data Types

This [blog](https://databricks.com/blog/2017/02/23/working-complex-data-formats-structured-streaming-apache-spark-2-1.html) is very useful.

It is common to have complex data types such as **structs, maps, and arrays** when working with semi-structured formats. 

### Selecting from nested columns

Dots (`.`) can be used to access nested columns for structs and maps.

```scala
// input
{
  "a": {
     "b": 1
  }
}

Python: events.select("a.b")
 Scala: events.select("a.b")
   SQL: select a.b from events

// output
{
  "b": 1
}
```

### Flattening structs

A star (`*`) can be used to select all of the subfields in a struct.

```scala
// input
{
  "a": {
     "b": 1,
     "c": 2
  }
}

Python:  events.select("a.*")
 Scala:  events.select("a.*")
   SQL:  select a.* from events

// output
{
  "b": 1,
  "c": 2
}
```

### Nesting columns

The struct function or just parentheses in SQL can be used to create a new struct.

```language-scala
// input
{
  "a": 1,
  "b": 2,
  "c": 3
}

Python: events.select(struct(col("a").alias("y")).alias("x"))
 Scala: events.select(struct('a as 'y) as 'x)
   SQL: select named_struct("y", a) as x from events

// output
{
  "x": {
    "y": 1
  }
}
```

### Selecting a single array or map element

`getItem()` or square brackets (i.e. `[ ]`) can be used to select a single element out of an array or a map.

```language-scala
// input
{
  "a": [1, 2]
}

Python: events.select(col("a").getItem(0).alias("x"))
 Scala: events.select('a.getItem(0) as 'x)
   SQL: select a[0] as x from events

// output
{ "x": 1 }
// input
{
  "a": {
    "b": 1
  }
}

Python: events.select(col("a").getItem("b").alias("x"))
 Scala: events.select('a.getItem("b") as 'x)
   SQL: select a['b'] as x from events

// output
{ "x": 1 }
```

### Creating a row for each array or map element

`explode()` can be used to create a new row for each element in an array or each key-value pair. This is similar to LATERAL VIEW EXPLODE in HiveQL.

```language-scala
// input
{
  "a": [1, 2]
}

Python: events.select(explode("a").alias("x"))
 Scala: events.select(explode('a) as 'x)
   SQL: select explode(a) as x from events

// output
[{ "x": 1 }, { "x": 2 }]
// input
{
  "a": {
    "b": 1,
    "c": 2
  }
}

Python: events.select(explode("a").alias("x", "y"))
 Scala: events.select(explode('a) as Seq("x", "y"))
   SQL: select explode(a) as (x, y) from events

// output
[{ "x": "b", "y": 1 }, { "x": "c", "y": 2 }]
```

### Collecting multiple rows into an array

`collect_list()` and `collect_set()` can be used to aggregate items into an array.

```language-scala
// input
[{ "x": 1 }, { "x": 2 }]

Python: events.select(collect_list("x").alias("x"))
 Scala: events.select(collect_list('x) as 'x)
   SQL: select collect_list(x) as x from events

// output
{ "x": [1, 2] }
// input
[{ "x": 1, "y": "a" }, { "x": 2, "y": "b" }]

Python: events.groupBy("y").agg(collect_list("x").alias("x"))
 Scala: events.groupBy("y").agg(collect_list('x) as 'x)
   SQL: select y, collect_list(x) as x from events group by y

// output
[{ "y": "a", "x": [1]}, { "y": "b", "x": [2]}]
```

### Selecting one field from each item in an array

When you use dot notation on an array we return a new array where that field has been selected from each array element.

```language-scala
// input
{
  "a": [
    {"b": 1},
    {"b": 2}
  ]
}

Python: events.select("a.b")
 Scala: events.select("a.b")
   SQL: select a.b from events

// output
{
  "b": [1, 2]
}
```

## Power of to_json() and from_json()

Spark SQL provides functions like `to_json()` to encode a struct as a string and `from_json()` to retrieve the struct as a complex type. Using JSON strings as columns are useful when reading from or writing to a streaming source like Kafka. Each Kafka key-value record will be augmented with some metadata, such as the ingestion timestamp into Kafka, the offset in Kafka, etc. If the “value” field that contains your data is in JSON, you could use `from_json()` to extract your data, enrich it, clean it, and then push it downstream to Kafka again or write it out to a file.

#### Encode a struct as json

`to_json()` can be used to turn structs into JSON strings. This method is particularly useful when you would like to re-encode multiple columns into a single one when writing data out to Kafka. This method is not presently available in SQL.

```language-scala
// input
{
  "a": {
    "b": 1
  }
}

Python: events.select(to_json("a").alias("c"))
 Scala: events.select(to_json('a) as 'c)

// output
{
  "c": "{\"b\":1}"
}
```

#### Decode json column as a struct

`from_json()` can be used to turn a string column with JSON data into a struct. Then you may flatten the struct as described above to have individual columns. This method is not presently available in SQL.

```language-scala
// input
{
  "a": "{\"b\":1}"
}

Python: 
  schema = StructType().add("b", IntegerType())
  events.select(from_json("a", schema).alias("c"))
Scala:
  val schema = new StructType().add("b", IntegerType)
  events.select(from_json('a, schema) as 'c)

// output
{
  "c": {
    "b": 1
  }
}
```

Sometimes you may want to leave a part of the JSON string still as JSON to avoid too much complexity in your schema.

```language-scala
// input
{
  "a": "{\"b\":{\"x\":1,\"y\":{\"z\":2}}}"
}

Python: 
  schema = StructType().add("b", StructType().add("x", IntegerType())
                              .add("y", StringType()))
  events.select(from_json("a", schema).alias("c"))
Scala:
  val schema = new StructType().add("b", new StructType().add("x", IntegerType)
    .add("y", StringType))
  events.select(from_json('a, schema) as 'c)

// output
{
  "c": {
    "b": {
      "x": 1,
      "y": "{\"z\":2}"
    }
  }
}
```

#### Parse a set of fields from a column containing JSON

`json_tuple()` can be used to extract fields available in a string column with JSON data.

```language-scala
// input
{
  "a": "{\"b\":1}"
}

Python: events.select(json_tuple("a", "b").alias("c"))
Scala:  events.select(json_tuple('a, "b") as 'c)
SQL:    select json_tuple(a, "b") as c from events

// output
{ "c": 1 }
```

Sometimes a string column may not be self-describing as JSON, but may still have a well-formed structure. For example, it could be a log message generated using a specific Log4j format. Spark SQL can be used to structure those strings for you with ease!

#### Parse a well-formed string column

`regexp_extract()` can be used to parse strings using regular expressions.

```language-scala
// input
[{ "a": "x: 1" }, { "a": "y: 2" }]

Python: events.select(regexp_extract("a", "([a-z]):", 1).alias("c"))
Scala:  events.select(regexp_extract('a, "([a-z]):", 1) as 'c)
SQL:    select regexp_extract(a, "([a-z]):", 1) as c from events

// output
[{ "c": "x" }, { "c": "y" }]
```

## More Json!

The useful functions are `get_json_object()`, `from_json()`, `to_json()`, `explode()` and `selectExpr()`

Checkout this [notebook](https://docs.databricks.com/_static/notebooks/complex-nested-structured.html) on databricks!!!

### A simple Json Schema without nested Structure

```scala
import org.apache.spark.sql.types._                         // include the Spark Types to define our schema
import org.apache.spark.sql.functions._                     // include the Spark helper functions

val jsonSchema = new StructType()
        .add("battery_level", LongType)
        .add("c02_level", LongType)
        .add("cca3",StringType)
        .add("cn", StringType)
        .add("device_id", LongType)
        .add("device_type", StringType)
        .add("signal", LongType)
        .add("ip", StringType)
        .add("temp", LongType)
        .add("timestamp", TimestampType)
```

### `get_json_object()`

`get_json_object()` extracts a JSON object from a JSON string based on JSON path specified, and returns a JSON string as the extracted JSON object. 

```scala
val jsDF = eventsFromJSONDF.select($"id", get_json_object($"json", "$.device_type").alias("device_type"),
                                          get_json_object($"json", "$.ip").alias("ip"),
                                         get_json_object($"json", "$.cca3").alias("cca3"))
```

This allows you to **extract json fields as columns**

### `from_json()`

`from_json()` is a variation of `get_json_object()`, this function **uses schema to extract individual columns**. Using `from_json()` helper function within the `select()` Dataset API call, we can extract or encode data's attributes and values from a JSON string into a DataFrame as columns, dictated by a schema.

In example below:

- Uses the schema above to extract from the JSON string attributes and values and represent them as individual columns as part of `devices`
- `select()` all its columns
- Filters on desired attributes using the `.` notation

Once you have extracted data from a JSON string into its respective DataFrame columns, you can apply DataFrame/Dataset APIs calls to select, filter, and subsequtly display, to your satisfaction.

```scala
val devicesDF = eventsDS.select(from_json($"device", jsonSchema) as "devices")
.select($"devices.*")
.filter($"devices.temp" > 10 and $"devices.signal" > 15)
```

### `to_json()` 

You can convert or encode our filtered devices into JSON string using `to_json()`. That is, **convert a JSON struct into a string.** 

```scala
val stringJsonDF = eventsDS.select(to_json(struct($"*"))).toDF("devices")

```

### `selectExpr()`

Another way to convert or encode a column into a JSON object as string is to use the *selectExpr()* utility function. For instance, I can convert the "device" column of our DataFrame from above into a JSON String

```scala
val stringsDF = eventsDS.selectExpr("CAST(id AS INT)", "CAST(device AS STRING)")
```

Another use of `selectExpr()` is its ability, as the function name suggests, **take expressions as arguments and convert them into respective columns**. For instance, say I want to express c02 levels and temperature ratios.

```scala
devicesDF.selectExpr("c02_level", "round(c02_level/temp) as ratio_c02_temperature").orderBy($"ratio_c02_temperature" desc)
```

### Nested Structure

It's not unreasonable to assume that your JSON nested structures may have Maps as well as nested JSON. For illustration, let's use a single string comprised of complex and nested data types, including a Map. In a real life scenario, this could be a reading from a device event, with dangerous levels of C02 emissions or high temperature readings, that needs Network Operation Center (NOC) notification for some immediate action.

```scala
import org.apache.spark.sql.types._

val schema = new StructType()
  .add("dc_id", StringType)                               // data center where data was posted to Kafka cluster
  .add("source",                                          // info about the source of alarm
    MapType(                                              // define this as a Map(Key->value)
      StringType,
      new StructType()
      .add("description", StringType)
      .add("ip", StringType)
      .add("id", LongType)
      .add("temp", LongType)
      .add("c02_level", LongType)
      .add("geo", 
         new StructType()
          .add("lat", DoubleType)
          .add("long", DoubleType)
        )
      )
    )
```

```scala
//create a single entry with id and its complex and nested data types

val dataDS = Seq("""
{
"dc_id": "dc-101",
"source": {
    "sensor-igauge": {
      "id": 10,
      "ip": "68.28.91.22",
      "description": "Sensor attached to the container ceilings",
      "temp":35,
      "c02_level": 1475,
      "geo": {"lat":38.00, "long":97.00}                        
    },
    "sensor-ipad": {
      "id": 13,
      "ip": "67.185.72.1",
      "description": "Sensor ipad attached to carbon cylinders",
      "temp": 34,
      "c02_level": 1370,
      "geo": {"lat":47.41, "long":-122.00}
    },
    "sensor-inest": {
      "id": 8,
      "ip": "208.109.163.218",
      "description": "Sensor attached to the factory ceilings",
      "temp": 40,
      "c02_level": 1346,
      "geo": {"lat":33.61, "long":-111.89}
    },
    "sensor-istick": {
      "id": 5,
      "ip": "204.116.105.67",
      "description": "Sensor embedded in exhaust pipes in the ceilings",
      "temp": 40,
      "c02_level": 1574,
      "geo": {"lat":35.93, "long":-85.46}
    }
  }
}""").toDS()
// should only be one item
dataDS.count()
```

```scala
val df = spark                  // spark session 
.read                           // get DataFrameReader
.schema(schema)                 // use the defined schema above and read format as JSON
.json(dataDS.rdd)               // RDD[String]
```

### `explode()`

The [`explode()`](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.functions$) function is used to show how to extract nested structures. Plus, it sheds more light when we see how it works alongside `to_json()` and `from_json()` functions, when extracting attributes and values from complex JSON structures. So on occasion, you will want to use `explode()`, alongside `to_json()` and `from_json()` functions. And here's one case where we do.

The `explode()` function creates a new row for each element in the given map column. In this case, the map column is `source`. Note that for each key-value in the map, you have a respective Row, in this case four.

```scala
val explodedDF = df.select($"dc_id", explode($"source"))
```

Now we can access the data from our exploded data using Map

```scala
//case class to denote our desired Scala object
case class DeviceAlert(dcId: String, deviceType:String, ip:String, deviceId:Long, temp:Long, c02_level: Long, lat: Double, lon: Double)
//access all values using getItem() method on value, by providing the "key," which is attribute in our JSON object.
val notifydevicesDS = explodedDF.select( $"dc_id" as "dcId",
                        $"key" as "deviceType",
                        'value.getItem("ip") as 'ip,
                        'value.getItem("id") as 'deviceId,
                        'value.getItem("c02_level") as 'c02_level,
                        'value.getItem("temp") as 'temp,
                        'value.getItem("geo").getItem("lat") as 'lat,                //note embedded level requires yet another level of fetching.
                        'value.getItem("geo").getItem("long") as 'lon)
                        .as[DeviceAlert]  // return as a Dataset
```

> Note that we use `getItem()` to retrieve a value in a map

# Working with complex type Columns

## Map Type Columns

### Fetching values from maps with `.` (dot)

```scala
singersDF
	.withColumn("song_to_love", $"songs.good_song")
	.show(false)
```

# A real Example

If your schema is wrong, there will **not** be exception thrown but the corresponding columns will be Null!!!! (Which is annoying but fundamentally due to the **untyped** nature of dataframe)

![截屏2021-07-20 下午6.37.38](%E6%88%AA%E5%B1%8F2021-07-20%20%E4%B8%8B%E5%8D%886.37.38.png)

对应解析代码：

```scala
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._                     // include the Spark helper functions
val aidRecallInfoSchema
 = new StructType()
    .add("ad_type", LongType)
    .add("dpa_id", LongType)
    .add("cids", 
        ArrayType (
            LongType, true
        )
    )
    .add("aid", LongType)
    .add("direct_id_infos", 
        MapType (
            StringType,
            new StructType()
            .add ("check_id_infos",
                MapType  (
                    StringType,
                    new StructType()
                    .add("through_count", LongType, true)
                    .add("step_count", LongType, true)
                    .add("pid_infos", 
                        ArrayType (
                            new StructType()
                            .add("product_id", LongType, true)
                            .add("recallInfo", 
                                new StructType()
                                .add("source_type", StringType, true)
                                .add("source_key", StringType, true)
                                .add("sim_score", FloatType, true)
                                , true)
                            .add("recommend_pkg_name", StringType, true)
                        ), true)
                )
            )
        )
    )
```

