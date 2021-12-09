---
title: Python a short manual
date: 2020-07
tags:
    - Python
    - Programming
---
# Pitfalls!!!

## Unexpectedly Appending Reference

The following code will produce unexpected result

```python
dict_a = {'a' : 1}
dict_list = [dict_a]
>>> dict_list 
>>> [{'a' : 1}]
dict_a['a'] = 2
>>> dict_list
>>> [{'a' : 2}]
```

That is, what's in the list of dictionary is just a reference, not an object. Make sure to use `dict_list.append(dict_a.copy())`  if you don't want this sort of behavior

Similarly, appending list to list incurs the same behavior.

> `.copy()` creates what's called a shallow copy, see [here](https://stackoverflow.com/questions/5105517/deep-copy-of-a-dict-in-python) for a more thorough discussion

The assignment operator `=` in Python does not create a copy, but bind the objects together. To get around that, we can use `copy`. However:

**The shallow copy creates a copy of the object, but references each element of the object.**

```python
old_list = [[1,2,3], [4,5,6]]
new_list = copy.copy(old_list) # shallow copy
cold_list[0][2] = 'c' 
>>> old_list
>>> [[1,2,c],[4,5,6]]
>>> new_list
>>> [[1,2,c],[4,5,6]]
```

> Shallow copy works fine if all the objects are immutable

 To get around the behavior above, we can use deep copy provided by `copy.deepcopy( obj )`. This will create copies of the objects that are hold by `obj`

# Virtual Environment

To create a virtual environment, decide upon a directory where you want to place it, and run the [`venv`](https://docs.python.org/3/library/venv.html#module-venv) module as a script with the directory path:

```bash
python3 -m venv tutorial-env
```

This will create a virtual environment called `tutorial-env`, to activate the virtual environment, use

```bash
source tutorial-env/bin/activate
(tutorial-env) $ ...
```



# Dictionary

## Advanced Methods

### `get()` method

The syntax is 

```python
dict.get(key [,value])
```

where `value` is returned when the `key` is not found

### `update()` method

```python
dict.update([other])
```

The `update()` method taks either a dictionary or an iterable object of key/value pairs (generally tuples). It adds the elements to the dictionary if the key is not in the dictionary. If the key is in the dictionary, it updates the key with the new value. **It does not return any values**

> Note that the `update()` method assign with **shallow copy**. If this is unwanted, you should call `b_dict.update(copy.deepcopy(a_dict))`

### `pop()` method

The Python pop() method removes an item from a dictionary.

This method can also be used to remove items from a list at a particular index value.

```python
dictionary.pop(key_to_remove, not_found)
```



# Iterator

`__iter__()` is called with the iterator is created using `iter()` or on initializing a for loop. It should include `return self` or other iterator with `__next__()` defined.  

The `__next__()` function must be implemented for an iterator. In a for loop, it is called until `StopIteration` is raised. Typically:  

	def __next__(self):
		if !end:
			...
			return value
		else:
			raise StopIteration

You can use `next()` to get the next element of an iterator.  

> Call next() not iter.next()

# List Comprehension

## Cartesian Product

	[(a,b) for a in listA for b in listB]



# OOP in Python

## Override built in methods

This defines a functor like object.

Override `__call__()` function:

```
class FunctionLike(object):
    def __call__(self, a):
        print("I got called with {!r}!".format(a))

fn = FunctionLike()
fn(10)

# --> I got called with 10!
```

## Abstract class

```python
class Foo:
    def __getitem__(self, index):
        ...
    def __len__(self):
        ...
    def get_iterator(self):
        return iter(self)

class MyIterable(ABC):

    @abstractmethod
    def __iter__(self):
        while False:
            yield None

    def get_iterator(self):
        return self.__iter__()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MyIterable:
            if any("__iter__" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented

MyIterable.register(Foo)
```

# Functional

## `functools.partial(fun, /, *args, **keywords)`

Return a new [partial object](https://docs.python.org/3/library/functools.html#partial-objects) which when called will behave like *func* called with the positional arguments *args* and keyword arguments *keywords*. If more arguments are supplied to the call, they are appended to *args*. If additional keyword arguments are supplied, they extend and override *keywords*. Roughly equivalent to:`def partial(func, /, *args, **keywords):    def newfunc(*fargs, **fkeywords):        newkeywords = {**keywords, **fkeywords}        return func(*args, *fargs, **newkeywords)    newfunc.func = func    newfunc.args = args    newfunc.keywords = keywords    return newfunc `The [`partial()`](https://docs.python.org/3/library/functools.html#functools.partial) is used for partial function application which “freezes” some portion of a function’s arguments and/or keywords resulting in a new object with a simplified signature. For example, [`partial()`](https://docs.python.org/3/library/functools.html#functools.partial) can be used to create a callable that behaves like the [`int()`](https://docs.python.org/3/library/functions.html#int) function where the *base* argument defaults to two:>>>`>>> from functools import partial >>> basetwo = partial(int, base=2) >>> basetwo.__doc__ = 'Convert base 2 string to an int.' >>> basetwo('10010') 18 `



# Datatime module

## Current date and time

	from datetime import datetime
	now = datetime.now()
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

## UNIX timestamp and datetime conversion

```python
import datetime
timestamp = datetime(year, month, day, hour, minute, second) # year etc. are all integers, convert to timestamp
datetime.datetime.fromtimestamp(timestamp) # return datetime
```

## string -> datetime

```python
dt = datetime.strptime(time_str, "%Y%m$d") # parse string like "20210721"
```

For formatting, checkout [reference from web](https://www.journaldev.com/23365/python-string-to-datetime-strptime)







# Logging

## Basics

	import logging
	logging.basicConfig(filename="fileanme",level=logging.INFO)
	logging.info('info msg')
	logging.warning('warning msg')
	logging.debug('debug msg')

# re (Regular Expression)

## Basics

You can compile a regular expression into a *pattern* object and then use various methods it provides. For example,  

	import re
	
	p = re.compile('[a-z]+')
	
	p.match(text) # if the start of the string matches
	m = p.search(text) # search through the string
	m = p.findall(text)
	m.finditer(text) # find all the substrings and returns them as an iterator


	m.group() # return the string matched by the RE
	m.start() # return the starting position of the re
	m.end() # return the ending positionn of the re
	m.span() # return a tuple containing the start, end positions of the match

## Restricting matched string length

`{n,m}` will match repetition at least `n` tiems but not exceeding `m` times.  

# Matplotlib

Matplotlib is a library for plotting. 

## Generic plotting function

Below is a generic plotting function `plot_on_ax()` . It takes an `ax` object and plot data, add legend as well as setting title, xlabel and ylabel.

```python
def plot_on_ax(
    data,
    ax,
    title=None, 
    xlabel=None, 
    ylabel=None, 
    legend=None,
    legend_loc='best',
    fontsize={},
    aux_func=[]
    ):
    line, = ax.plot(data)
    if title:
        ax.set_title(title, fontsize=fontsize.get('title',16))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize.get('xlabel',12))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize.get('ylabel',12))
    ax.tick_params(axis='both', which='major', labelsize=fontsize.get('tick', 8))
    for func_ in aux_func:
        ax = func_(ax)
    if legend:
        line.set_label(legend)
        ax.legend(loc=legend_loc, prop={'size': fontsize.get('legend', 6)}, ncol=2)
    return ax

# a working example
from functools import partial
def plot_horizontal(ax, y, color='black', lstyle='--', legend=None):
    line = ax.axhline(y=y, color=color, linestyle=lstyle)
    if legend:
        line.set_label(legend)
    return ax
plot_h = partial(plot_horizontal, y=thresh, legend=f'Threshold={thresh}')

fig = plt.figure(figsize=(24,8))
plt.tight_layout()
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

# plotting
p1_idx, p2_idx = 0, 15
plot_on_ax(np.abs(automat[p1_idx,:]), ax1, title=f'Auto-correlation for player {p1_idx}',xlabel='Samples', ylabel='Auto-correlation', legend='(abs) Auto-correlation', aux_func=[plot_h])
plot_on_ax(np.abs(automat[p2_idx,:]), ax2, title=f'Auto-correlation for player {p2_idx}',xlabel='Samples', ylabel='Auto-correlation', legend='(abs) Auto-correlation', aux_func=[plot_h])
# saving 
fig.savefig('fig/fig2.png', dpi=80)
```

## Annotating with horizontal / vertical lines

```python
# Horizontal line
def plot_horizontal(ax, y, color='black', lstyle='--', legend=None):
    line = ax.axhline(y=y, color=color, linestyle=lstyle)
    if legend:
        line.set_label(legend)
    return ax
plot_h = partial(plot_horizontal, y=thresh, legend=f'Threshold={thresh}')

# Vertical line
def plot_vertical(ax, x, color='black', lstyle='--', legend=None):
    line = ax.axvline(x=x, color=color, linestyle=lstyle)
    if legend:
        line.set_label(legend)
    return ax
burn_in = 100
plot_v = partial(plot_vertical, x=burn_in, legend=f'Burn_in={burn_in}', color='r')
```

## Ploting horizontal barplots

```python
def plot_ranking(skills, xlabel=None, title=None):
    fig = plt.figure(figsize=(16,36))
    player_skills = [(p, w) for [p], w in zip(W, skills)]
    sorted_skills = sorted(player_skills, key=lambda x: x[1], reverse=True)
    plt.barh(range(len(sorted_skills)), [v for _, v in sorted_skills])
    plt.yticks(range(len(sorted_skills)),labels=[p for p, _ in sorted_skills], fontsize=16)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel, fontsize=20) if xlabel else None
    plt.ylabel('Player Name', fontsize=20)
    plt.title(title, fontsize=20)
    # plt.xticks(range(len(sorted_skills)),labels=[p for p, _ in sorted_skills], rotation=90)
    plt.show()
    return sorted_skills, fig
```



## Working Examples

	fig, ax = plt.subplots(2) # row_num, column_num, figsize=(x,y)
	fig.tight_layout()
	
	data = np.random.rand(1000)
	x = np.linspace(-3, 3, 100)
	y = x**2
	
	ax[0].hist(data, bins=100, density=True)
	ax[0].set_xlabel('...')
	ax[0].set_title('Gaussian Histogram')
	
	ax[1].plot(x, y, label="quadratic")
	ax[1].set_ylim(0, 3)
	
	plt.legend()
	plt.show()

## Setting titles

	fig.suptitle('title', fontsize=16, y=1)
	ax[0].set_title('...')

## Saving figures

	plt.savefig('plot.png', dpi=300, bbox_inches='tight')

Note that this must be called **before** `plt.show()`. Otherwise you will get a blank image.

# Pandas

Refer to the pandas handbook for details

## Basics

### Creating DataFrame and Series

	import pandas as pd
	df = pd.DataFrame({'col1_name':[v1,v2,v3,...], 'col2_name':[v1,v2,v3,...],index=['1','2',...])
	series = pd.Series([v1,v2,v3,...],index=['ind1','ind2',...])

### Appending records

You can append another dataframe to existing one.

	df = df.append(df2, ignore_index=True)

If the columns are named `A` and `B`, you may use:  

	df = df.append({'A':value,'B':value},ignore_index=True)

To append a record, note the you need to **ASSIGN** back the new appended df.  

### Reading/Saving Data

	df = pd.read_csv('path_to_file', index_col=0)
	df.to_csv('path_to_file')

### Chunk write/read

If the csv file is large, you can read the csv file in chunks with the `chucksize` argument  
	

	for chunk in pd.read_csv('large.csv', chunksize=100): #reads 100 rows each time
		process(chunk) # chunk is a DataFrame object 

To write a large csv file, use the following:
	

	df.to_csv(filename, header=None, mode='a')

Note that you need to let `header = True` (default) for the first batch.  

## Indexing

### Naive Indexing

Just like with common python indexing, you may treat `df` as a fancy dictonary. Slicing works in the same way as well.  

### `loc` and `iloc`

`loc` allows indexing with labels. For example, `df.loc['col_name']` is allowed. Whereas `iloc` is only **index-based**.  
You can specify which columns to choose by conditioning in the `[]` operator. For example:  
		

		df.loc[df.country=='China']

Returns a series with country name China. You can also specify a range of values using `isin()`, for example:  
		

		df.loc[df.country.isin(['China','Japan'])]

You can also use `&` or `|` to form compound predicate. For example:  
		

		df.loc[(df.country=='China') & (df.id==1)]

Note that the parantheses is **compulsory**. `&` stands for AND, `|` stands for OR.  

### Iterating over DataFrame

	for index, row in df.iterrows():
		process(row)
		...

## Tricks

### `drop` to drop columns/rows

	skip_list = [1,2,3,4]
	df = pd.read_csv('filename')
	df.drop(skip_list, erros='ignore')

`errors` argument can be helpful for handling batch data, index not in list error will be ignored and only existing indices are dropped. Using `axis=1` to remove columns.  

## Cautions

### Assign column to column

You need to do 

```python
res_df[bad_case_by] = anno_df[bad_case_by].tolist()
```

the `tolist()` method is important. Otherwise the column might be NaN.

# json

## Basics

JSON (JavaScript Object Notation) is a popular data format to represent structured data. In python, JSON exists as a **string**. For example: 

	import json
	p = '{"name": "Bob", "languages": ["Python", "Java"]}'

### **Json string -> dict**  

	dict = json.loads(json_str)

### **Json file -> dict**

	with open(filename,'r') as f:
		data = json.load(f)

### **dict -> JSON string/file**

	json_str = json.dumps(dict)
	with open(filename, 'w') as f:
		json.dump(dict, f)

### **Pretty print JSON**



	py_dict = json.loads(json_str)
	print(json.dumps(person_dict, indent = 4, sort_keys=True))

# Numpy

## Random Number Generation 

### Random Integer

#### np.random.choice(*a*, *size=None*, *replace=True*, *p=None*)

Generates a random sample from a given 1-D array

*New in version 1.7.0.*

> New code should use the `choice` method of a `default_rng()` instance instead; please see the [Quick Start](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start).

- Parameters

  **a**: 1-D array-like or intIf an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)**size**int or tuple of ints, optionalOutput shape. If the given shape is, e.g., `(m, n, k)`, then `m * n * k` samples are drawn. Default is None, in which case a single value is returned.

  **replace**: boolean, optionalWhether the sample is with or without replacement

  **p**: 1-D array-like, optionalThe probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.

- Returns

  **samples** single item or ndarrayThe generated random samples

- Raises

  ValueError If a is an int and less than zero, if a or p are not 1-dimensional, if a is an array-like of size 0, if p is not a vector of probabilities, if a and p have different lengths, or if replace=False and the sample size is greater than the population size

### np.where(condition, [,x,y])

Return elements chosen from *x* or *y* depending on *condition*.

>  When only *condition* is provided, this function is a shorthand for `np.asarray(condition).nonzero()`. Using [`nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero) directly should be preferred, as it behaves correctly for subclasses. The rest of this documentation covers only the case where all three arguments are provided.

- Parameters

  **condition **: array_like, bool Where True, yield *x*, otherwise yield *y*.

  **x, y**: array_like Values from which to choose. *x*, *y* and *condition* need to be broadcastable to some shape.

- Returns

  **out** ndarray An array with elements from *x* where *condition* is True, and elements from *y* elsewhere.

### Boolean Mask

Boolean mask can be used to select based on some condition

```python
A[A[:,0]==1] # select all rows whose first element equals 1
```



# Fast Fourier Transform

## Inspecting Spectrum from Samples

```
psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd)) # fs is the sampling frequency
plt.plot(f, psd)
plt.show()
```

# Fuzzy

The code below shows how to do fuzzy matching with python

```python
!pip install fuzzywuzzy
!pip install python-Levenshtein
from fuzzywuzzy import fuzz

fuzz.token_sort_ratio(str1, str2) / 100.0 > 0.9
```



# Syntax Sugar

## Unpacking (star operator *)

The single star `*` unpacks the sequence/collection into positional arguments, so you can do this:

```python
def sum(a, b):
    return a + b

values = (1, 2)

s = sum(*values)
```

This will unpack the tuple so that it actually executes as:

```python
s = sum(1, 2)
```

The double star `**` does the same, only using a dictionary and thus named arguments:

```python
values = { 'a': 1, 'b': 2 }
s = sum(**values)
```

