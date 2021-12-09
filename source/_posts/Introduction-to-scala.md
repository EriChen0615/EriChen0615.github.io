---
title: Introduction to scala
date: 2021-07
tags:
    - Scala
    - Programming
---
# Introduction

## What is Scala?

Scala is a modern multi-paradigm programming language designed to express common programming patterns in a concise, elegant, and type-safe way. It seamlessly integrates features of object-oriented and functional languages.

## Scala is object-oriented

Scala is a **pure object-oriented language** in the sense that [every value is an object](https://docs.scala-lang.org/tour/unified-types.html). Types and behaviors of objects are described by [classes](https://docs.scala-lang.org/tour/classes.html) and [traits](https://docs.scala-lang.org/tour/traits.html). Classes can be extended by subclassing, and by using a flexible [mixin-based composition](https://docs.scala-lang.org/tour/mixin-class-composition.html) mechanism as a clean replacement for multiple inheritance.

## Scala is functional

Scala is also **a functional language** in the sense that [every function is a value](https://docs.scala-lang.org/tour/unified-types.html). Scala provides a [lightweight syntax](https://docs.scala-lang.org/tour/basics.html#functions) for defining anonymous functions, it supports [higher-order functions](https://docs.scala-lang.org/tour/higher-order-functions.html), it allows functions to be [nested](https://docs.scala-lang.org/tour/nested-functions.html), and it supports [currying](https://docs.scala-lang.org/tour/multiple-parameter-lists.html). Scala’s [case classes](https://docs.scala-lang.org/tour/case-classes.html) and its built-in support for [pattern matching](https://docs.scala-lang.org/tour/pattern-matching.html) provide the functionality of algebraic types, which are used in many functional languages. [Singleton objects](https://docs.scala-lang.org/tour/singleton-objects.html) provide a convenient way to group functions that aren’t members of a class.

Furthermore, Scala’s notion of pattern matching naturally extends to the [processing of XML data](https://github.com/scala/scala-xml/wiki/XML-Processing) with the help of [right-ignoring sequence patterns](https://docs.scala-lang.org/tour/regular-expression-patterns.html), by way of general extension via [extractor objects](https://docs.scala-lang.org/tour/extractor-objects.html). In this context, [for comprehensions](https://docs.scala-lang.org/tour/for-comprehensions.html) are useful for formulating queries. These features make Scala ideal for developing applications like web services.

## Scala is statically typed

Scala’s expressive type system enforces, at compile-time, that abstractions are used in a safe and coherent manner. In particular, the type system supports:

- [Generic classes](https://docs.scala-lang.org/tour/generic-classes.html)
- [Variance annotations](https://docs.scala-lang.org/tour/variances.html)
- [Upper](https://docs.scala-lang.org/tour/upper-type-bounds.html) and [lower](https://docs.scala-lang.org/tour/lower-type-bounds.html) type bounds
- [Inner classes](https://docs.scala-lang.org/tour/inner-classes.html) and [abstract type members](https://docs.scala-lang.org/tour/abstract-type-members.html) as object members
- [Compound types](https://docs.scala-lang.org/tour/compound-types.html)
- [Explicitly typed self references](https://docs.scala-lang.org/tour/self-types.html)
- [Implicit parameters](https://docs.scala-lang.org/tour/implicit-parameters.html) and [conversions](https://docs.scala-lang.org/tour/implicit-conversions.html)
- [Polymorphic methods](https://docs.scala-lang.org/tour/polymorphic-methods.html)

[Type inference](https://docs.scala-lang.org/tour/type-inference.html) means the user is not required to annotate code with redundant type information. In combination, these features provide a powerful basis for the safe reuse of programming abstractions and for the type-safe extension of software.

## Scala is extensible

In practice, the development of domain-specific applications often requires domain-specific language extensions. Scala provides a unique combination of language mechanisms that make it straightforward to add new language constructs in the form of libraries.

In many cases, this can be done without using meta-programming facilities such as macros. For example:

- [Implicit classes](https://docs.scala-lang.org/overviews/core/implicit-classes.html) allow adding extension methods to existing types.
- [String interpolation](https://docs.scala-lang.org/overviews/core/string-interpolation.html) is user-extensible with custom interpolators.

## Scala interoperates

Scala is designed to interoperate well with the popular Java Runtime Environment (JRE). In particular, the interaction with the mainstream object-oriented Java programming language is as seamless as possible. Newer Java features like SAMs, [lambdas](https://docs.scala-lang.org/tour/higher-order-functions.html), [annotations](https://docs.scala-lang.org/tour/annotations.html), and [generics](https://docs.scala-lang.org/tour/generic-classes.html) have direct analogues in Scala.

Those Scala features without Java analogues, such as [default](https://docs.scala-lang.org/tour/default-parameter-values.html) and [named parameters](https://docs.scala-lang.org/tour/named-arguments.html), compile as closely to Java as reasonably possible. Scala has the same compilation model (separate compilation, dynamic class loading) as Java and allows access to thousands of existing high-quality libraries.

# Scala Basics

## Values and Variables

You can name the results of expressions using the `val` keyword.

```scala
val x = 1 + 1
println(x) // 2
x = 3 // This does not compile
```

**Values cannot be reassigned.** The type of a value can be omitted and inferred, or explicitly stated

```scala
val x: Int = 1 + 1 
```

> However, if `val` defines an object of a class, its mutable members defined by `var` can be modified. 
>
> ```scala
> class Point(var x: Int, var y: Int)
> val pt1 = new Point(1, 1)
> pt1.x = 2 // This is valid!
> ```

Variables are like values, except you can re-assign them. You can define a variable with the `var` keyword

```scala
var x = 1 + 1;
x = 3 // valid
println(x*x)
```

## Blocks

You can combine expressions by surrounding them with `{}`. We call this a block.

**The result of the last expression in the block is the result of the overall block**

```scala
println({
  val x = 1 + 1
  x + 1
}) // 3
```

## Functions

unctions are expressions that have parameters, and take arguments.

You can define an anonymous function (i.e., a function that has no name) that returns a given integer plus one:

```scala
(x: Int) => x + 1
```

On the left of `=>` is a list of parameters. On the right is an expression involving the parameters.

You can also name functions:

```scala
val addOne = (x: Int) => x + 1
println(addOne(1)) // 2
```

A function can have multiple parameters:

```scala
val add = (x: Int, y: Int) => x + y
println(add(1, 2)) // 3
```

Or it can have no parameters at all:

```scala
val getTheAnswer = () => 42
println(getTheAnswer()) // 42
```

> The syntax is similar to the JavaScript's arrow function.

## Methods

Methods look and behave very similar to functions, but there are a few key differences between them.

Methods are defined with the `def` keyword. `def` is followed by a name, parameter list(s), a return type, and a body:

```scala
def add(x: Int, y: Int): Int = x + y
println(add(1, 2)) // 3
```

Notice how the return type `Int` is declared *after* the parameter list and a `:`.

**A method can take multiple parameter lists:**

```scala
def addThenMultiply(x: Int, y: Int)(multiplier: Int): Int = (x + y) * multiplier
println(addThenMultiply(1, 2)(3)) // 9
```

> Notice how the function is called with multiple parameter lists

Or no parameter lists at all:

```scala
def name: String = System.getProperty("user.name")
println("Hello, " + name + "!")
```

## Classes

You can define classes with the `class` keyword, followed by its name and constructor parameters:

```scala
class Greeter(prefix: String, suffix: String) {
  def greet(name: String): Unit =
    println(prefix + name + suffix)
}
```

T**he return type of the method `greet` is `Unit`, which signifies that there is nothing meaningful to return. It is used similarly to `void` in Java and C.** (A difference is that, because every Scala expression must have some value, there is actually a singleton value of type Unit, written (). It carries no information.)

You can make an instance of a class with the `new` keyword:

```scala
val greeter = new Greeter("Hello, ", "!")
greeter.greet("Scala developer") // Hello, Scala developer!
```

## Case Classes

Scala has a special type of class called a “case” class. By default, instances of case classes are immutable, and they are compared by value (unlike classes, whose instances are compared by reference). This makes them additionally useful for [pattern matching](https://docs.scala-lang.org/tour/pattern-matching.html#matching-on-case-classes).

You can define case classes with the `case class` keywords:

```scala
case class Point(x: Int, y: Int)
```

You can instantiate case classes without the `new` keyword:

```scala
val point = Point(1, 2)
val anotherPoint = Point(1, 2)
val yetAnotherPoint = Point(2, 2)
```

Instances of case classes are compared by value, not by reference:

```scala
if (point == anotherPoint) {
  println(point + " and " + anotherPoint + " are the same.")
} else {
  println(point + " and " + anotherPoint + " are different.")
} // Point(1,2) and Point(1,2) are the same.

if (point == yetAnotherPoint) {
  println(point + " and " + yetAnotherPoint + " are the same.")
} else {
  println(point + " and " + yetAnotherPoint + " are different.")
} // Point(1,2) and Point(2,2) are different.
```

> Note that if the Point is a class and its objects initialized with the `new` keyword, `point `will not equal `anotherPoint` because we are now comparing the addresses. Case classes is like an auto-implementation of the equal `=` operator with element-wise comparison 

## Objects

Objects are single instances of their own definitions. You can think of them as singletons of their own classes.

You can define objects with the `object` keyword:

```scala
object IdFactory {
  private var counter = 0
  def create(): Int = {
    counter += 1
    counter
  }
}
```

You can access an object by referring to its name:

```scala
val newId: Int = IdFactory.create()
println(newId) // 1
val newerId: Int = IdFactory.create()
println(newerId) // 2
```

## Traits

Traits are abstract data types containing certain fields and methods. In Scala inheritance, a class can only extend one other class, but it can extend multiple traits.

You can define traits with the `trait` keyword:

```scala
trait Greeter {
  def greet(name: String): Unit
}
```

Traits can also have default implementations:

```scala
trait Greeter {
  def greet(name: String): Unit =
    println("Hello, " + name + "!")
}
```

You can extend traits with the `extends` keyword and override an implementation with the `override` keyword:

```scala
class DefaultGreeter extends Greeter

class CustomizableGreeter(prefix: String, postfix: String) extends Greeter {
  override def greet(name: String): Unit = {
    println(prefix + name + postfix)
  }
}

val greeter = new DefaultGreeter()
greeter.greet("Scala developer") // Hello, Scala developer!

val customGreeter = new CustomizableGreeter("How are you, ", "?")
customGreeter.greet("Scala developer") // How are you, Scala developer?
```

## Main Method

The main method is the entry point of a Scala program. The Java Virtual Machine requires a main method, named `main`, that takes one argument: an array of strings.

Using an object, you can define the main method as follows:

```scala
object Main {
  def main(args: Array[String]): Unit =
    println("Hello, Scala developer!")
}
```

## infix syntax and dot-notation

The following codes are equivalent

```scala
true && true  //infix
true.&&(true) //dot-notation
```

# Unified Types

In Scala, all values have a type, including numerical values and functions. The diagram below illustrates a subset of the type hierarchy.

![Scala Type Hierarchy](unified-types-diagram.svg)

[`Any`](https://www.scala-lang.org/api/2.12.1/scala/Any.html) is the supertype of all types, also called the top type. It defines certain universal methods such as `equals`, `hashCode`, and `toString`. `Any` has two direct subclasses: `AnyVal` and `AnyRef`.

`AnyVal` represents value types. There are nine predefined value types and they are non-nullable: `Double`, `Float`, `Long`, `Int`, `Short`, `Byte`, `Char`, `Unit`, and `Boolean`. `Unit` is a value type which carries no meaningful information. There is exactly one instance of `Unit` which can be declared literally like so: `()`. All functions must return something so sometimes `Unit` is a useful return type.

`AnyRef` represents reference types. All non-value types are defined as reference types. Every user-defined type in Scala is a subtype of `AnyRef`. If Scala is used in the context of a Java runtime environment, `AnyRef` corresponds to `java.lang.Object`.

Here is an example that demonstrates that strings, integers, characters, boolean values, and functions are all of type `Any` just like every other object:

```scala
val list: List[Any] = List(
  "a string",
  732,  // an integer
  'c',  // a character
  true, // a boolean value
  () => "an anonymous function returning a string"
)

list.foreach(element => println(element))
```

## Type Casting

Value types can be cast in the following way: [![Scala Type Hierarchy](https://docs.scala-lang.org/resources/images/tour/type-casting-diagram.svg)](https://docs.scala-lang.org/resources/images/tour/type-casting-diagram.svg)

For example:

```scala
val x: Long = 987654321
val y: Float = x  // 9.8765434E8 (note that some precision is lost in this case)

val face: Char = '☺'
val number: Int = face  // 9786
```

Casting is unidirectional. This will not compile:

```scala
val x: Long = 987654321
val y: Float = x  // 9.8765434E8
val z: Long = y  // Does not conform
```

You can also cast a reference type to a subtype. This will be covered later in the tour.

## Nothing and Null

`Nothing` is a subtype of all types, also called the bottom type. There is no value that has type `Nothing`. A common use is to signal non-termination such as a thrown exception, program exit, or an infinite loop (i.e., it is the type of an expression which does not evaluate to a value, or a method that does not return normally).

`Null` is a subtype of all reference types (i.e. any subtype of AnyRef). It has a single value identified by the keyword literal `null`. `Null` is provided mostly for interoperability with other JVM languages and **should almost never be used in Scala code.** We’ll cover alternatives to `null` later in the tour.

# Classes

## Constructors

Constructors can have optional parameters by providing a default value like so:

```scala
class Point(var x: Int = 0, var y: Int = 0)

val origin = new Point  // x and y are both set to 0
val point1 = new Point(1)
println(point1.x)  // prints 1
```

In this version of the `Point` class, `x` and `y` have the default value `0` so no arguments are required. However, because the constructor reads arguments left to right, if you just wanted to pass in a `y` value, you would need to name the parameter.

```scala
class Point(var x: Int = 0, var y: Int = 0)
val point2 = new Point(y = 2)
println(point2.y)  // prints 2
```

## Private Members and Getter/Setter Syntax

Members are public by default. Use the `private` access modifier to hide them from outside of the class.

```scala
class Point {
  private var _x = 0
  private var _y = 0
  private val bound = 100

  def x = _x
  def x_= (newValue: Int): Unit = {
    if (newValue < bound) _x = newValue else printWarning
  }

  def y = _y
  def y_= (newValue: Int): Unit = {
    if (newValue < bound) _y = newValue else printWarning
  }

  private def printWarning = println("WARNING: Out of bounds")
}

val point1 = new Point
point1.x = 99
point1.y = 101 // prints the warning
```

In this version of the `Point` class, the data is stored in private variables `_x` and `_y`. There are methods `def x` and `def y` for accessing the private data. `def x_=` and `def y_=` are for validating and setting the value of `_x` and `_y`. Notice the special syntax for the setters: the method has `_=` appended to the identifier of the getter and the parameters come after.

Primary constructor parameters with `val` and `var` are public. However, because `val`s are immutable, you can’t write the following.

```scala
class Point(val x: Int, val y: Int)
val point = new Point(1, 2)
point.x = 3  // <-- does not compile
```

Parameters without `val` or `var` are private values, visible only within the class.

```scala
class Point(x: Int, y: Int)
val point = new Point(1, 2)
point.x  // <-- does not compile
```

# Traits 

## Defining a trait

A minimal trait is simply the keyword `trait` and an identifier:

```scala
trait HairColor
```

Traits become especially useful as generic types and with abstract methods.

```scala
trait Iterator[A] {
  def hasNext: Boolean
  def next(): A
}
```

Extending the `trait Iterator[A]` requires a type `A` and implementations of the methods `hasNext` and `next`.

## Using traits

Use the `extends` keyword to extend a trait. Then implement any abstract members of the trait using the `override` keyword:

```scala
trait Iterator[A] {
  def hasNext: Boolean
  def next(): A
}

class IntIterator(to: Int) extends Iterator[Int] {
  private var current = 0
  override def hasNext: Boolean = current < to
  override def next(): Int = {
    if (hasNext) {
      val t = current
      current += 1
      t
    } else 0
  }
}


val iterator = new IntIterator(10)
iterator.next()  // returns 0
iterator.next()  // returns 1
```

This `IntIterator` class takes a parameter `to` as an upper bound. It `extends Iterator[Int]` which means that the `next` method must return an Int.

## Subtyping

Where a given trait is required, a subtype of the trait can be used instead.

```scala
import scala.collection.mutable.ArrayBuffer

trait Pet {
  val name: String
}

class Cat(val name: String) extends Pet
class Dog(val name: String) extends Pet

val dog = new Dog("Harry")
val cat = new Cat("Sally")

val animals = ArrayBuffer.empty[Pet]
animals.append(dog)
animals.append(cat)
animals.foreach(pet => println(pet.name))  // Prints Harry Sally
```

The `trait Pet` has an abstract field `name` that gets implemented by Cat and Dog in their constructors. On the last line, we call `pet.name`, which must be implemented in any subtype of the trait `Pet`.

# Tuples

In Scala, a tuple is a value that contains a fixed number of elements, each with its own type. Tuples are immutable.

**Tuples are especially handy for returning multiple values from a method.**

A tuple with two elements can be created as follows:

```scala
val ingredient = ("Sugar" , 25)
```

This creates a tuple containing a `String` element and an `Int` element.

The inferred type of `ingredient` is `(String, Int)`, which is shorthand for `Tuple2[String, Int]`.

To represent tuples, Scala uses a series of classes: `Tuple2`, `Tuple3`, etc., through `Tuple22`. Each class has as many type parameters as it has elements.

## Accessing the elements

One way of accessing tuple elements is by position. The individual elements are named `_1`, `_2`, and so forth.

```scala
println(ingredient._1) // Sugar
println(ingredient._2) // 25
```

## Pattern matching on tuples

A tuple can also be taken apart using pattern matching:

```scala
val (name, quantity) = ingredient
println(name) // Sugar
println(quantity) // 25
```

Here `name`’s inferred type is `String` and `quantity`’s inferred type is `Int`.

Here is another example of pattern-matching a tuple:

```scala
val planets =
  List(("Mercury", 57.9), ("Venus", 108.2), ("Earth", 149.6),
       ("Mars", 227.9), ("Jupiter", 778.3))
planets.foreach{
  case ("Earth", distance) =>
    println(s"Our planet is $distance million kilometers from the sun")
  case _ =>
}
```

Or, in `for` comprehension:

```scala
val numPairs = List((2, 5), (3, -7), (20, 56))
for ((a, b) <- numPairs) {
  println(a * b)
}
```

## Tuples and case classes

Users may sometimes find it hard to choose between tuples and case classes. Case classes have named elements. The names can improve the readability of some kinds of code. In the planet example above, we might define `case class Planet(name: String, distance: Double)` rather than using tuples.

# Class Composition with Mixins

Mixins are traits which are used to compose a class.

```scala
abstract class A {
  val message: String
}
class B extends A {
  val message = "I'm an instance of class B"
}
trait C extends A {
  def loudMessage = message.toUpperCase()
}
class D extends B with C

val d = new D
println(d.message)  // I'm an instance of class B
println(d.loudMessage)  // I'M AN INSTANCE OF CLASS B
```

Class `D` has a superclass `B` and a mixin `C`. Classes can only have one superclass but many mixins (using the keywords `extends` and `with` respectively). The mixins and the superclass may have the same supertype.

Now let’s look at a more interesting example starting with an abstract class:

```scala
abstract class AbsIterator {
  type T
  def hasNext: Boolean
  def next(): T
}
```

The class has an abstract type `T` and the standard iterator methods.

Next, we’ll implement a concrete class (all abstract members `T`, `hasNext`, and `next` have implementations):

```scala
class StringIterator(s: String) extends AbsIterator {
  type T = Char
  private var i = 0
  def hasNext = i < s.length
  def next() = {
    val ch = s charAt i
    i += 1
    ch
  }
}
```

`StringIterator` takes a `String` and can be used to iterate over the String (e.g. to see if a String contains a certain character).

Now let’s create a trait which also extends `AbsIterator`.

```scala
trait RichIterator extends AbsIterator {
  def foreach(f: T => Unit): Unit = while (hasNext) f(next())
}
```

This trait implements `foreach` by continually calling the provided function `f: T => Unit` on the next element (`next()`) as long as there are further elements (`while (hasNext)`). Because `RichIterator` is a trait, it doesn’t need to implement the abstract members of AbsIterator.

We would like to combine the functionality of `StringIterator` and `RichIterator` into a single class.

```scala
class RichStringIter extends StringIterator("Scala") with RichIterator
val richStringIter = new RichStringIter
richStringIter.foreach(println)
```

The new class `RichStringIter` has `StringIterator` as a superclass and `RichIterator` as a mixin.

With single inheritance we would not be able to achieve this level of flexibility.

# Higher-order Functions

Higher order functions take other functions as parameters or return a function as a result. This is possible because functions are first-class values in Scala. The terminology can get a bit confusing at this point, and we use the phrase “higher order function” for both methods and functions that take functions as parameters or that return a function.

One of the most common examples is the higher-order function `map` which is available for collections in Scala.

```scala
val salaries = Seq(20000, 70000, 40000)
val doubleSalary = (x: Int) => x * 2
val newSalaries = salaries.map(doubleSalary) // List(40000, 140000, 80000)
```

`doubleSalary` is a function which takes a single Int, `x`, and returns `x * 2`. In general, the tuple on the left of the arrow `=>` is a parameter list and the value of the expression on the right is what gets returned. On line 3, the function `doubleSalary` gets applied to each element in the list of salaries.

To shrink the code, we could make the function anonymous and pass it directly as an argument to map:

```scala
val salaries = Seq(20000, 70000, 40000)
val newSalaries = salaries.map(x => x * 2) // List(40000, 140000, 80000)
```

Notice how `x` is **not declared as an Int** in the above example. That’s because the compiler can infer the type based on the type of function map expects (see [Currying](https://docs.scala-lang.org/tour/multiple-parameter-lists.html). An even more idiomatic way to write the same piece of code would be:

```scala
val salaries = Seq(20000, 70000, 40000)
val newSalaries = salaries.map(_ * 2)
```

>  Since the Scala compiler already knows the type of the parameters (a single Int), you just need to provide the right side of the function. The only caveat is that you need to use `_` in place of a parameter name (it was `x` in the previous example).

## Coercing methods into functions

It is also possible to pass methods as arguments to higher-order functions because the Scala compiler will coerce the method into a function.

```scala
case class WeeklyWeatherForecast(temperatures: Seq[Double]) {

  private def convertCtoF(temp: Double) = temp * 1.8 + 32

  def forecastInFahrenheit: Seq[Double] = temperatures.map(convertCtoF) // <-- passing the method convertCtoF
}
```

Here the method `convertCtoF` is passed to the higher order function `map`. This is possible because the compiler coerces `convertCtoF` to the function `x => convertCtoF(x)` (note: `x` will be a generated name which is guaranteed to be unique within its scope).

## Functions that accept functions

One reason to use higher-order functions is to reduce redundant code. Let’s say you wanted some methods that could raise someone’s salaries by various factors. Without creating a higher-order function, it might look something like this:

```scala
object SalaryRaiser {

  def smallPromotion(salaries: List[Double]): List[Double] =
    salaries.map(salary => salary * 1.1)

  def greatPromotion(salaries: List[Double]): List[Double] =
    salaries.map(salary => salary * math.log(salary))

  def hugePromotion(salaries: List[Double]): List[Double] =
    salaries.map(salary => salary * salary)
}
```

Notice how each of the three methods vary only by the multiplication factor. To simplify, you can extract the repeated code into a higher-order function like so:

```scala
object SalaryRaiser {

  private def promotion(salaries: List[Double], promotionFunction: Double => Double): List[Double] =
    salaries.map(promotionFunction)

  def smallPromotion(salaries: List[Double]): List[Double] =
    promotion(salaries, salary => salary * 1.1)

  def greatPromotion(salaries: List[Double]): List[Double] =
    promotion(salaries, salary => salary * math.log(salary))

  def hugePromotion(salaries: List[Double]): List[Double] =
    promotion(salaries, salary => salary * salary)
}
```

The new method, `promotion`, takes the salaries plus a function of type `Double => Double` (i.e. a function that takes a Double and returns a Double) and returns the product.

Methods and functions usually express behaviours or data transformations, therefore having functions that compose based on other functions can help building generic mechanisms. Those generic operations defer to lock down the entire operation behaviour giving clients a way to control or further customize parts of the operation itself.

## Functions that return functions

There are certain cases where you want to generate a function. Here’s an example of a method that returns a function.

```scala
def urlBuilder(ssl: Boolean, domainName: String): (String, String) => String = {
  val schema = if (ssl) "https://" else "http://"
  (endpoint: String, query: String) => s"$schema$domainName/$endpoint?$query"
}

val domainName = "www.example.com"
def getURL = urlBuilder(ssl=true, domainName)
val endpoint = "users"
val query = "id=1"
val url = getURL(endpoint, query) // "https://www.example.com/users?id=1": String
```

Notice the return type of urlBuilder `(String, String) => String`. This means that the returned anonymous function takes two Strings and returns a String. In this case, the returned anonymous function is `(endpoint: String, query: String) => s"https://www.example.com/$endpoint?$query"`.

# Multiple Parameter Lists (Currying)

### Example

Here is an example, as defined on the `Iterable` trait in Scala’s collections API:

```scala
trait Iterable[A] {
  ...
  def foldLeft[B](z: B)(op: (B, A) => B): B
  ...
}
```

`foldLeft` applies a two-parameter function `op` to an initial value `z` and all elements of this collection, going left to right. Shown below is an example of its usage.

Starting with an initial value of 0, `foldLeft` here applies the function `(m, n) => m + n` to each element in the List and the previous accumulated value.

```scala
val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
val res = numbers.foldLeft(0)((m, n) => m + n)
println(res) // 55
```

### Use cases

Suggested use cases for multiple parameter lists include:

#### DRIVE TYPE INFERENCE

It so happens that in Scala, type inference proceeds one parameter list at a time. Say you have the following method:

```scala
def foldLeft1[A, B](as: List[A], b0: B, op: (B, A) => B) = ???
```

Then you’d like to call it in the following way, but will find that it doesn’t compile:

```scala
def notPossible = foldLeft1(numbers, 0, _ + _)
```

you will have to call it like one of the below ways:

```scala
def firstWay = foldLeft1[Int, Int](numbers, 0, _ + _)
def secondWay = foldLeft1(numbers, 0, (a: Int, b: Int) => a + b)
```

That’s because Scala won’t be able to infer the type of the function `_ + _`, as it’s still inferring `A` and `B`. By moving the parameter `op` to its own parameter list, `A` and `B` are inferred in the first parameter list. These inferred types will then be available to the second parameter list and `_ + _` will match the inferred type `(Int, Int) => Int`

```scala
def foldLeft2[A, B](as: List[A], b0: B)(op: (B, A) => B) = ???
def possible = foldLeft2(numbers, 0)(_ + _)
```

#### IMPLICIT PARAMETERS

To specify only certain parameters as [`implicit`](https://docs.scala-lang.org/tour/implicit-parameters.html), they must be placed in their own `implicit` parameter list.

An example of this is:

```scala
def execute(arg: Int)(implicit ec: scala.concurrent.ExecutionContext) = ???
```

#### PARTIAL APPLICATION

When a method is called with a fewer number of parameter lists, then this will yield a function taking the missing parameter lists as its arguments. This is formally known as [partial application](https://en.wikipedia.org/wiki/Partial_application).

For example,

```scala
val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
val numberFunc = numbers.foldLeft(List[Int]()) _

val squares = numberFunc((xs, x) => xs :+ x*x)
println(squares) // List(1, 4, 9, 16, 25, 36, 49, 64, 81, 100)

val cubes = numberFunc((xs, x) => xs :+ x*x*x)
println(cubes)  // List(1, 8, 27, 64, 125, 216, 343, 512, 729, 1000)
```

# Case Classes

## Defining a case class

A minimal case class requires the keywords `case class`, an identifier, and a parameter list (which may be empty):

```scala
case class Book(isbn: String)

val frankenstein = Book("978-0486282114")
```

Notice how the keyword `new` was not used to instantiate the `Book` case class. This is because case classes have an `apply` method by default which takes care of object construction.

> When you create a case class with parameters, the parameters are public `val`s.

```scala
case class Message(sender: String, recipient: String, body: String)
val message1 = Message("guillaume@quebec.ca", "jorge@catalonia.es", "Ça va ?")

println(message1.sender)  // prints guillaume@quebec.ca
message1.sender = "travis@washington.us"  // this line does not compile
```

You can’t reassign `message1.sender` because it is a `val` (i.e. immutable). It is possible to use `var`s in case classes but this is discouraged.

## Comparison

Instances of case classes are compared by structure and not by reference:

```scala
case class Message(sender: String, recipient: String, body: String)

val message2 = Message("jorge@catalonia.es", "guillaume@quebec.ca", "Com va?")
val message3 = Message("jorge@catalonia.es", "guillaume@quebec.ca", "Com va?")
val messagesAreTheSame = message2 == message3  // true
```

Even though `message2` and `message3` refer to different objects, the value of each object is equal.

## Copying

You can create a (shallow) copy of an instance of a case class simply by using the `copy` method. You can optionally change the constructor arguments.

```scala
case class Message(sender: String, recipient: String, body: String)
val message4 = Message("julien@bretagne.fr", "travis@washington.us", "Me zo o komz gant ma amezeg")
val message5 = message4.copy(sender = message4.recipient, recipient = "claire@bourgogne.fr")
message5.sender  // travis@washington.us
message5.recipient // claire@bourgogne.fr
message5.body  // "Me zo o komz gant ma amezeg"
```

The recipient of `message4` is used as the sender of `message5` but the `body` of `message4` was copied directly.

# Pattern Matching

Pattern matching is a mechanism for checking a value against a pattern. A successful match can also deconstruct a value into its constituent parts. It is a more powerful version of the `switch` statement in Java and it can likewise be used in place of a series of if/else statements.

## Syntax

A match expression has a value, the `match` keyword, and at least one `case` clause.

```scala
import scala.util.Random

val x: Int = Random.nextInt(10)

x match {
  case 0 => "zero"
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}
```

The `val x` above is a random integer between 0 and 10. `x` becomes the left operand of the `match` operator and on the right is an expression with four cases. The last case `_` is a “catch all” case for any other possible `Int` values. Cases are also called *alternatives*.

## Matching on case classes

Case classes are especially useful for pattern matching.

```scala
abstract class Notification

case class Email(sender: String, title: String, body: String) extends Notification

case class SMS(caller: String, message: String) extends Notification

case class VoiceRecording(contactName: String, link: String) extends Notification
```

`Notification` is an abstract super class which has three concrete Notification types implemented with case classes `Email`, `SMS`, and `VoiceRecording`. Now we can do pattern matching on these case classes:

```scala
def showNotification(notification: Notification): String = {
  notification match {
    case Email(sender, title, _) =>
      s"You got an email from $sender with title: $title"
    case SMS(number, message) =>
      s"You got an SMS from $number! Message: $message"
    case VoiceRecording(name, link) =>
      s"You received a Voice Recording from $name! Click the link to hear it: $link"
  }
}
val someSms = SMS("12345", "Are you there?")
val someVoiceRecording = VoiceRecording("Tom", "voicerecording.org/id/123")

println(showNotification(someSms))  // prints You got an SMS from 12345! Message: Are you there?

println(showNotification(someVoiceRecording))  // prints You received a Voice Recording from Tom! Click the link to hear it: voicerecording.org/id/123
```

## Pattern guards

Pattern guards are simply boolean expressions which are used to make cases more specific. Just add `if <boolean expression>` after the pattern.

```scala
def showImportantNotification(notification: Notification, importantPeopleInfo: Seq[String]): String = {
  notification match {
    case Email(sender, _, _) if importantPeopleInfo.contains(sender) =>
      "You got an email from special someone!"
    case SMS(number, _) if importantPeopleInfo.contains(number) =>
      "You got an SMS from special someone!"
    case other =>
      showNotification(other) // nothing special, delegate to our original showNotification function
  }
}

val importantPeopleInfo = Seq("867-5309", "jenny@gmail.com")

val someSms = SMS("123-4567", "Are you there?")
val someVoiceRecording = VoiceRecording("Tom", "voicerecording.org/id/123")
val importantEmail = Email("jenny@gmail.com", "Drinks tonight?", "I'm free after 5!")
val importantSms = SMS("867-5309", "I'm here! Where are you?")

println(showImportantNotification(someSms, importantPeopleInfo)) // prints You got an SMS from 123-4567! Message: Are you there?
println(showImportantNotification(someVoiceRecording, importantPeopleInfo)) // prints You received a Voice Recording from Tom! Click the link to hear it: voicerecording.org/id/123
println(showImportantNotification(importantEmail, importantPeopleInfo)) // prints You got an email from special someone!

println(showImportantNotification(importantSms, importantPeopleInfo)) // prints You got an SMS from special someone!
```

In the `case Email(sender, _, _) if importantPeopleInfo.contains(sender)`, the pattern is matched only if the `sender` is in the list of important people.

## Matching on type only

You can match on the type like so:

```scala
abstract class Device
case class Phone(model: String) extends Device {
  def screenOff = "Turning screen off"
}
case class Computer(model: String) extends Device {
  def screenSaverOn = "Turning screen saver on..."
}

def goIdle(device: Device) = device match {
  case p: Phone => p.screenOff
  case c: Computer => c.screenSaverOn
}
```

`def goIdle` has a different behavior depending on the type of `Device`. This is useful when the case needs to call a method on the pattern. It is a convention to use the first letter of the type as the case identifier (`p` and `c` in this case).

# FOR Comprehension

cala offers a lightweight notation for expressing *sequence comprehensions*. Comprehensions have the form `for (enumerators) yield e`, where `enumerators` refers to a semicolon-separated list of enumerators. An *enumerator* is either a generator which introduces new variables, or it is a filter. A comprehension evaluates the body `e` for each binding generated by the enumerators and returns a sequence of these values.

Here’s an example:

```scala
case class User(name: String, age: Int)

val userBase = List(
  User("Travis", 28),
  User("Kelly", 33),
  User("Jennifer", 44),
  User("Dennis", 23))

val twentySomethings =
  for (user <- userBase if user.age >=20 && user.age < 30)
  yield user.name  // i.e. add this to a list

twentySomethings.foreach(println)  // prints Travis Dennis
```

Here is a more complicated example using two generators. It computes all pairs of numbers between `0` and `n-1` whose sum is equal to a given value `v`:

```scala
def foo(n: Int, v: Int) =
   for (i <- 0 until n;
        j <- 0 until n if i + j == v)
   yield (i, j)

foo(10, 10) foreach {
  case (i, j) =>
    println(s"($i, $j) ")  // prints (1, 9) (2, 8) (3, 7) (4, 6) (5, 5) (6, 4) (7, 3) (8, 2) (9, 1)
}
```

