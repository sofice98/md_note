JavaSE：标准版（桌面程序，控制台）
JavaME：嵌入式开发（手机）
JavaEE：E企业级开发（web，服务器）



JDK(Java Development Kit)：整个java开发工具
JRE(Java Runtime Environment)：运行时环境
JVM(Java Virtual Machine)：java虚拟机

Java 程序其实是运行在JVM (Java虚拟机) 上的，使用 Java 编译器编译 Java 程序时，生成的是与平台无关的字节码，**这些字节码只面向 JVM**。不同平台的 JVM 都是不同的，但它们都提供了相同的接口，这也正是 Java 跨平台的原因。

普通用户只需要安装 JRE 来运行 Java 程序。而程序开发者必须安装 JDK 来编译、调试程序

![image-20210304101524025](..\Resources\jdk.png)

JDK目录：

- `bin` 文件里面存放了JDK的各种开发工具的可执行文件，主要的是编译器 (`javac.exe`)
- `db` 文件是一个先进的全事务处理的基于 Java 技术的数据库（jdk 自带数据库 db 的使用）
- `include` 文件里面是 Java 和 JVM 交互用的头文件
- `jre` 为 Java 运行环境
- `lib` 文件存放的是 JDK 工具命令的实际执行程序

JRE目录：

- `bin` 里的就是 JVM
- `lib` 中则是 JVM 工作所需要的类库



源代码的文件名必须与文件中公共类 public class 的名字相同。







# 基本类型

**8中基本类型**使用频率高，存在栈中

基本类型**默认值**仅在 Java 初始化类的时候才会被赋予，局部变量不会

Java 没有 sizeof

- **整型 byte(1) / short(2) / int(4) / long(8)**

  在 Java 中， 整型的范围与运行 Java 代码的机器无关（平台无关性）。必须保证在所有机器上都能够得到相同的运行结果， 所以各种数据类型的取值范围必须固定，也没有 sizeof。

  长整型：后缀L	十六进制：前缀0x	二进制：前缀0b

  数字字面量加下划线（如用 1_000_000 表示一百万），这些下划线只是为了提高可读性，Java 编译器会去除这些下划线

  Java **没有任何无符号（unsigned) 形式**的 int、 long、short 或 byte 类型。

- **浮点型 float(4) / double(8)**

  float：后缀f	double：后缀d（或默认）

  ```java
//正无穷大
  Double_POSITIVE_INFINITY
  //负无穷大
  Double.NEGATIVEJNFINITY
  //NaN (不是一个数字）  
  Double.NaN
  //检测非数字
  if(Double.isNaN(x))
  ```
  
- **boolean类型(1/8)**

  整型值、整数表达式和布尔值之间不能进行相互转换

- **char型(2)**



![图片](..\Resources\类型合法转换.jpg)

**高精度数值 BigInteger / BigDecimal**

可以处理包含任意长度数字序列的数值

```java
import java.math.BigInteger;
import java.math.BigDecimal;
//将普通数值转化为大数值
BigInteger a = BigInteger.valueOf(100);
Biglnteger c = a.add(b); // c = a + b
Biglnteger d = c.multiply(b.add(Biglnteger.valueOf(2))); // d = c * (b + 2)
```



## 内置包装类

在 Java 中，万物皆对象，所有的操作都要求用对象的形式进行描述，为了把基本类型转换成对象，Java 给我们提供了完善的内置包装类

【Java 内置的包装类无法被继承】

| 基本类型 | 对应的包装类（位于 java.lang 包中） |
| :------: | :---------------------------------: |
|   byte   |                Byte                 |
|  short   |                Short                |
|   int    |               Integer               |
|   long   |                Long                 |
|  float   |                Float                |
|  double  |               Double                |
|   char   |              Character              |
| boolean  |               Boolean               |

前 6 个类派生于公共的超类 `Number`，而 `Character` 和 `Boolean` 是 `Object` 的直接子类

- **装箱**：将基本数据类型转换成包装类（每个包装类的构造方法都可以接收各自数据类型的变量）
- **拆箱**：从包装类之中取出被包装的基本类型数据（使用包装类的 xxxValue 方法）

```java
// JDK 1.5 之后
// 自动装箱，基本数据类型 int -> 包装类 Integer
// 等价于 Integer obj = Integer.valueOf(10);
Integer obj = 10;   
// 自动拆箱，Integer -> int
// 等价于 int temp = obj.intValue();
int temp = obj;   

obj ++; // 直接利用包装类的对象进行数学计算
System.out.println(temp * obj); 
```

【Integer.valueOf】：先判断 i 是否在缓存范围（默认 [-128, 127]）内，若在，则返回缓存池中对象（每次地址相同）；若不在，则创建新对象（每次new，地址不同）。因此不能用 `==` 判断值相等，要用`equals()` 方法

```java
// Integer.valueOf源码
public static Integer valueOf(int i) {
    if (i >= IntegerCache.low && i <= IntegerCache.high)
        return IntegerCache.cache[i + (-IntegerCache.low)];
    return new Integer(i);
}
```

【Object 类可以接收所有数据类型】：

```java
Object obj = 10;
int temp = (Integer) obj;
```

<img src="..\Resources\Object 类可以接收所有数据类型.png" alt="图片" style="zoom:75%;" />



## Object 通用方法

```java
public native int hashCode()
public boolean equals(Object obj)
protected native Object clone() throws CloneNotSupportedException
public String toString()
public final native Class<?> getClass()
protected void finalize() throws Throwable {}
public final native void notify()
public final native void notifyAll()
public final native void wait(long timeout) throws InterruptedException
public final void wait(long timeout, int nanos) throws InterruptedException
public final void wait() throws InterruptedException
```

1. **equals**

   基本类型比较可用 `==`，**对象比较最好用`n1.equals(n2)`**，该方法用于判断两个对象是否具有相同的引用（地址）

   1）：类没有覆盖 `equals()` 方法。则通过 `equals()` 比较该类的两个对象时，等价于通过 `==` 比较这两个对象（比较的是地址）。

   2）：类覆盖了 `equals()` 方法。一般来说，我们都覆盖 `equals()` 方法来判断两个对象的内容是否相等，比如 `String` 类。

2. **hashcode**

   hashCode() **返回哈希值**。

   在覆盖 equals() 方法时应当总是覆盖 hashCode() 方法，保证等价的两个对象哈希值也相等。

   HashSet  和 HashMap 等集合类使用了 hashCode()  方法来计算对象应该存储的位置，因此要将对象添加到这些集合类中，需要让对应的类实现 hashCode()  方法。

   下面的代码中，新建了两个等价的对象，并将它们添加到 HashSet 中。我们希望将这两个对象当成一样的，只在集合中添加一个对象。但是  EqualExample 没有实现 hashCode() 方法，因此这两个对象的哈希值是不同的，最终导致集合添加了两个等价的对象。

   ```java
   EqualExample e1 = new EqualExample(1, 1, 1);
   EqualExample e2 = new EqualExample(1, 1, 1);
   System.out.println(e1.equals(e2)); // true
   HashSet<EqualExample> set = new HashSet<>();
   set.add(e1);
   set.add(e2);
   System.out.println(set.size());   // 2
   ```

   理想的哈希函数应当具有均匀性，即不相等的对象应当均匀分布到所有可能的哈希值上。这就要求了哈希函数要把所有域的值都考虑进来。可以将每个域都当成 R 进制的某一位，然后组成一个 R 进制的整数。

   R 一般取 31，因为它是一个奇素数，如果是偶数的话，当出现乘法溢出，信息就会丢失，因为与 2 相乘相当于向左移一位，最左边的位丢失。并且一个数与 31 相乘可以转换成移位和减法：`31*x == (x<<5)-x`，编译器会自动进行这个优化。

   ```java
   @Override
   public int hashCode() {
       int result = 17;
       result = 31 * result + x;
       result = 31 * result + y;
       result = 31 * result + z;
       return result;
   }
   ```

3. **toString**

   默认返回 ToStringExample@4554617c 这种形式，其中 @ 后面的数值为散列码的无符号十六进制表示。

   重载使用valueOf：

   ```java
   class ToStringExample {
       private int number;
       @Override
       public String toString() {
           return String.valueOf(number);
       }
   }
   ```

4. **clone**

   使用 clone() 方法来拷贝一个对象即复杂又有风险，它会抛出异常，并且还需要类型转换

   可以使用拷贝构造函数或者拷贝工厂来拷贝一个对象

   ```java
   public class CloneConstructorExample {
   	public CloneConstructorExample(CloneConstructorExample original) {
           arr = new int[original.arr.length];
           for (int i = 0; i < original.arr.length; i++) {
               arr[i] = original.arr[i];
           }
       }
   }
   ```





# 字符串

**String**

在 Java 8 中，String 内部是使用 char 数组来存储数据的；而在 Java 9 之后，String 类的实现改用 byte 数组。不过，无论是 Java 8 还是 Java 9，用来存储数据的 char 或者 byte 数组 value 都一直是被声明为 final 的，这意味着 **String中value 数组初始化之后就不能再改变了**，并且 String内部没有改变 value 数组的方法。所有 String 的更变都是新建对象：

```java
String a = "hello";
String b = "world";
// 等价于 a = a + b
StringBuilder builder = new StringBuilder();
builder.append(a);
builder.append(b);
a = builder.toString();
// null拼接后会转化"null"
```

**空串与null**

```java
// 空串是一个 Java 对象， 有自己的串长度（0）和内容（空）
if(str.length() == 0)
// null，目前没有任何对象与该变量关联
if(str == null)
```

**任何一个 Java 对象都可以转换成字符串**：`String.valueOf(number);`

JVM 为了提高性能和减少内存开销，在实例化字符串常量的时候进行了一些**优化**：为字符串开辟了一个**【字符串常量池 String Pool】**，可以理解为缓存区。创建字符串常量时，首先检查字符串常量池中是否存在该字符串，若字符串常量池中存在该字符串，则直接返回该引用实例，无需重新实例化；若不存在，则实例化该字符串并放入池中。

JDK 1.7 之前，字符串常量池存在于【常量存储（Constant storage）】中；JDK 1.7 之后，字符串常量池存在于【堆内存（Heap）】中

```java
String str1 = "hello"; // 分配到常量池中
String str2 = new String(“hello”); // 先在String Pool 中开辟地址空间创建一个字符串对象，指向这个 "hello" 字符串字面量，然后在堆中创建一个字符串对象，使引用指向堆中的对象
String str3 = str2.intern(); // 如果 String Pool 中已经存在一个字符串和该字符串的值相等，那么就会返回 String Pool 中字符串的引用；否则，就会在 String Pool 中添加一个新的字符串，并返回这个新字符串的引用
```

【将String型数据变为基本数据类型】：

```java
String str = "10";
int temp = Integer.parseInt(str);// String -> int
```



**不可变的好处**

1. 可以缓存 hash 值：如 String 用做 HashMap 的 key，不可变的特性可以使得 hash 值也不可变，因此只需要进行一次计算
2. String Pool 的需要
3. 安全性
4. 线程安全：StringBuilder 不是线程安全的；StringBuffer 是线程安全的，内部使用 synchronized 进行同步



# 数组

```java
int[] a = {1, 2, 3};
int[] b = new int[3] {1, 2, 3};
double[][] b = new double[4][4];
```

Java 实际上没有多维数组，只有一维数组，多维数组被解释为【数组的数组】

```java
// 由于可以单独地存取数组的某一行， 所以可以让两行交换
int[] temp = b[1];
b[1] = b[2];
b[2] = temp;
// 使用 for each 循环遍历数组
for(int[] row : a) { // 遍历每一行
  for(int value : row) { // 遍历每一列
   System.out.println(value);
  }
}
```



**Arrays 类**

```java
import java.util.Arraysl;
// 将一维数组转成字符串类型（打印一维数组的所有元素）
Arrays.toString(a);
// 将二维数组转成字符串类型（打印二维数组的所有元素）
Arrays.deepToString(a);
// 数组拷贝，第 2 个参数是新数组的长度
Arrays.copyOf(a, 2 * a.length());
// 对数组中的元素进行排序
Arrays.sort(a);
// 比较数组，相等的条件是元素个数和对应位置的元素都相等
Arrays.equals(a, b);
```



**ArrayList**

可在运行过程中扩展数组的大小，但效率比数组低很多

集合不允许存储基本类型的数据，只能存储引用类型的数据

```java
ArrayList<Integer> list = new ArrayList<>();
```





# 集合

集合的主要功能：

- 存储不确定数量的数据（可以动态改变集合长度）
- 存储具有映射关系的数据
- 存储不同类型的数据

【集合只能存储引用类型（对象）】，如果存储的是 `int`型数据（基本类型），它会被自动装箱成 `Integer` 类型；而数组既可以存储基本数据类型，也可以存储引用类型

## Collection 接口

【单列集合】 `java.util.Collection`：元素是孤立存在的，向集合中存储元素采用一个个元素的方式存储。

<img src="..\Resources\collection.jpg" style="zoom:50%;" />

`Collection` 接口中定义了一些单列集合通用的方法：

```java
public boolean add(E e); // 把给定的对象添加到当前集合中
public void clear(); // 清空集合中所有的元素
public boolean remove(E e); // 把给定的对象在当前集合中删除
public boolean contains(E e); // 判断当前集合中是否包含给定的对象
public boolean isEmpty(); // 判断当前集合是否为空
public int size(); // 返回集合中元素的个数
public Object[] toArray(); // 把集合中的元素，存储到数组中
```

1. **List**

   【元素有序、可重复】

   - **ArrayList**：基于【动态数组】实现，支持随机访问。

   - **Vector**：和 ArrayList 类似，但它是线程安全的。

   - **LinkedList**：基于【双向链表】实现，只能顺序访问，但是可以快速地在链表中间插入和删除元素。不仅如此，LinkedList 还可以用作栈、队列和双向队列。

   `List` 接口而且还增加了一些根据元素索引来操作集合的特有方法：

   ```java
   public void add(int index, E element); // 将指定的元素，添加到该集合中的指定位置上
   public E get(int index); // 返回集合中指定位置的元素
   public E remove(int index); // 移除列表中指定位置的元素, 返回的是被移除的元素
   public E set(int index, E element); // 用指定元素替换集合中指定位置的元素
   ```

2. **Queue**

   - **LinkedList**：可以用它来实现双向队列。

   - **PriorityQueue**：基于【堆】结构实现，可以用它来实现优先队列。

3. **Set**

   【拒绝添加重复元素，不能通过整数索引来访问，元素无序】

   - **HashSet**：基于【HashMap 哈希表】实现，支持快速查找，但不支持有序性操作。并且失去了元素的插入顺序信息，也就是说使用 Iterator 遍历 HashSet 得到的结果是不确定的。

   - **LinkedHashSet**：底层是通过 【LinkedHashMap】来实现，具有 HashSet 的查找效率，并且内部使用双向链表维护元素的插入顺序。
   - **TreeSet**：基于【红黑树】实现，支持有序性操作，例如根据一个范围查找元素的操作。但是查找效率不如 HashSet，HashSet 查找的时间复杂度为 O(1)，TreeSet 则为 O(logN)。



## Map 接口

【双列集合】`java.util.Map` ：`Map` 不能包含重复的键，值可以重复；并且每个键只能对应一个值

<img src="E:\Programming\Github\md_note\..\Resources\map.jpg" style="zoom:50%;" />

`Map` 接口中定义了一些双列集合通用的方法：

```java
public V put(K key, V value); // 把指定的键与指定的值添加到 Map 集合中
public V remove(Object key); // 把指定的键所对应的键值对元素在 Map 集合中删除，返回被删除元素的值
public V get(Object key); // 根据指定的键，在 Map 集合中获取对应的值
boolean containsKey(Object key); // 判断集合中是否包含指定的键
public Set<K> keySet(); // 获取 Map 集合中所有的键，存储到 Set 集合中
// Entry 对象
public Set<Map.Entry<K,V>> entrySet(); // 获取 Map 中所有的 Entry 对象的集合
public K getKey(); // 获取某个 Entry 对象中的键
public V getValue(); // 获取某个 Entry 对象中的值
```

- HashMap：基于【哈希表】实现。
- LinkedHashMap：使用【双向链表】来维护元素的顺序，顺序为插入顺序或者最近最少使用（LRU）顺序。
- HashTable：和 HashMap 类似，但它是线程安全的，这意味着同一时刻多个线程同时写入 HashTable  不会导致数据不一致。它是遗留类，不应该去使用它，而是使用 ConcurrentHashMap 来支持线程安全，ConcurrentHashMap 的效率会更高，因为 ConcurrentHashMap 引入了分段锁。
- TreeMap：基于【红黑树】实现。



## 迭代器 Iterator

```java
// collection 这一表达式必须是一个数组或者是一个实现了 Iterable 接口的类对象
for(variable : collection) {
    // todo
}
public E next(); // 返回迭代的下一个元素。
public boolean hasNext(); // 如果仍有元素可以迭代，则返回 true
```

【为什么迭代器不封装成一个类，而是做成一个接口】：`Collection` 接口有很多不同的实现类，这些类的底层数据结构大多是不一样的，因此，它们各自的存储方式和遍历方式也是不同的，所以我们不能用一个类来规定死遍历的方法。



## 适配器

`java.util.Arrays#asList()` 可以把数组类型转换为 List 类型

```java
public static <T> List<T> asList(T... a)
// 不能使用基本类型数组作为参数，只能使用相应的包装类型数组。
Integer[] arr = {1, 2, 3};
List list = Arrays.asList(arr);
List list = Arrays.asList(1, 2, 3);
```





# 函数

 Java 程序设计语言总是采用按值调用。一个方法不能修改一个基本数据类型的参数，但可以修改引用参数

**方法签名=方法名字+参数列表**，返回类型和访问权限不是签名的一部分

```java
// 可变参数
public static int getSum(int... arr) {}
```





# I/O

```java
import java.util.Scanner;
import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
//输入对象
Scanner in = Scanner(System.in);
//按行
String name = in.nextLine();
//读单词
String name = in.next();
//读整数
int age = in.nextInt();
//输出
System.out.println("input");
System.out.printf("Hello,%s,Next year, you will be %d",name,age);
//文件I/O
Scanner in = new Scanner(Paths.get("myfile.txt"),"UTF-8");
PrintWriter out = new PrintWriter("file.txt","UTF-8");
out.printf("str");
out.close();
```



# 内存管理

## 数据存储

`String s;` 创建的只是引用，并不是对象。如果此时对 s 应用 String 方法，会报错。因为此时 s 没有与任何事物相关联。因此，一种安全的做法是：创建一个引用的同时便进行初始化

**栈**：存放Java 的**对象引用（变量名）和基本数据类型**。`int a = 3;`：编译器首先会在栈中创建一个变量名为 a 的引用，然后查找有没有字面值为 3 的地址，没找到，就在栈中开辟一个地址存放 3 这个字面值，然后将引用 a 指向 3 的地址。Java 系统必须知道存储在栈内的所有项的确切生命周期，以便上下移动指针。这一约束限制了程序的灵活性，所以 Java 对象并不存储在此。

**堆**：存放所有的 Java **对象**（**new**出来都存在堆中）。编译器不需要知道存储的数据在堆里存活多长时间

 **常量存储**：存放字符串常量和基本类型常量 `public static final`，这样做是安全的，因为它们永远不会被改变。

## 作用域

作用域由花括号 `{ }`的位置决定

但在 C/C++ 中，将一个较大作用域的变量隐藏的做法，在 Java 里是不允许的：

```java
{
    int x = 12;
    {
        int x = 123; // 非法
    }
}
```

当用 `new` 创建一个 Java 对象时，它可以存活于作用域之外。对象的引用 `s` 在作用域终点就消失了。然而，`s` 指向的 `String`对象仍占据内存空间。我们无法在作用域之后访问这个对象，因为对他唯一的引用已经超出了作用域的范围：

```java
{
    String s = new String("aas");
}
```

## 静态机制

1. 静态变量

   - 静态变量：又称为类变量，也就是说这个变量属于类的，类所有的实例都共享静态变量，可以直接通过类名来访问它。静态变量在内存中只存在一份。

   - 实例变量：每创建一个实例就会产生一个实例变量，它与该实例同生共死。

2. 静态方法

   静态方法在类加载的时候就存在了，它不依赖于任何实例。所以**静态方法必须实现**，也就是说它不能是抽象方法。而且**只能访问所属类的静态字段和静态方法**，方法中不能有 this 和 super 关键字，因此这两个关键字与具体对象关联。

   静态机制允许我们无需创建对象就可以直接通过类的引用来调用该方法，**使用类名直接引用静态变量或方法**是首选方案，因为它强调了静态属性

    ```java
    class StaticTest {
        static int i = 1;
        static void increment() { 
           this.i++; 
        }
    }
    //共享相同变量i
    StaticTest st1 = new StaticTest();
    StaticTest st2 = new StaticTest();
    //可以通过类名直接引用
    StaticTest.i++;
    StaticTest t = new StaticTest();
    StaticTest.increment();
    System.out.println(t.i);	//输出2
    ```

3. 静态语句块

   静态语句块在类初始化时运行一次。

4. 静态内部类

   非静态内部类依赖于外部类的实例，也就是说需要先创建外部类实例，才能用这个实例去创建非静态内部类。而静态内部类不需要。

   





## final关键字

1. 数据

   声明数据为常量，可以是编译时常量，也可以是在运行时被初始化后不能被改变的常量。

   - 对于基本类型，final 使数值不变；

   - 对于引用类型，final 使引用不变，也就不能引用其它对象，但是被引用的对象本身是可以修改的。

    ```java
    final int x = 1;
    // x = 2;  // cannot assign value to final variable 'x'
    final A y = new A();
    y.a = 1;
    ```

2. 方法

   声明方法不能被子类重写。

   private 方法隐式地被指定为 final，如果在子类中定义的方法和基类中的一个 private 方法签名相同，此时子类的方法不是重写基类方法，而是在子类中定义了一个新的方法。

3. 类

   声明类不允许被继承。





## 垃圾回收

Java 有一个**垃圾回收器**，用来监视 `new`创建的所有对象，并辨别那些不会被再引用的对象，然后释放这些对象的内存空间。

**finalize()**：当使用了内存之外的其他资源时使用，确保释放实例占用的全部资源。当垃圾收集器认为没有指向对象实例的引用时，会在销毁该对象之前调用 `finalize()` 方法。不过，在实际应用中，不要依赖于使用 `finalize`方法回收任何短缺的资源， 这是因为 **Java 并不保证定时为对象实例调用该方法，甚至不保证方法会被调用**，所以该方法不应该用于正常内存处理。





# 异常处理

异常处理框架基于基类`java.lang.throwable`，分为【错误 Error】和【异常 Exception】

Error：表示 JVM 无法处理的错误

Exception 分为两种：

- **受检异常**  ：需要用 try...catch... 语句捕获并进行处理，并且可以从异常中恢复；
- **非受检异常**  ：是程序运行时错误，例如除 0 会引发 Arithmetic Exception，此时程序崩溃并且无法恢复。

<img src="..\Resources\异常框架.png" alt="img" style="zoom: 15%;" />

整数被 0 除将会产生一个异常， 而浮点数被 0 除将会得到无穷大或 NaN 结果





# 面向对象编程

## 初始化

```java
class Test extends Cookie{
    private int birthday;
    private String sex;
    private int i;
    
    // 静态初始化块
    static{
        System.out.println("Root的静态初始化块");
    }
 	// 非静态初始化块
    {
        System.out.println("Root的普通初始化块");
    }
    // 默认构造函数
    Test(){ }
	// 无参构造函数
    public Test(){ }
	// 有参构造函数
    public Test(int birthday, String sex){ 
        //先初始化父类
        super(age);
        this.birthday = birthday;
        this.sex = sex;
    }
    
}

```

一旦你显式地定义了构造器（无论有参还是无参），编译器就不会自动为你创建无参构造器

**初始化块**

首先运行初始化块，然后才运行构造函数的主体部分

**初始化顺序**：

1. 静态变量和静态初始化块：使用 static 定义代码块，只有当类装载到系统时执行一次，之后不再执行。在静态初始化块中仅能初始化 static 修饰的数据成员
2. 实例变量和非静态初始化块
3. 构造函数



## 访问控制

**包（package）**

用来汇聚一组类， `package` 语句必须是文件中除了注释之外的第一行代码。



**访问修饰符（access specifier）**

用于修饰被封装的类的访问权限，从“最大权限”到“最小权限”依次是：

- 公开的 - `public`
- 受保护的 - `protected`：除了提供包访问权限，而且即使父类和子类不在同一个包下，子类也可以访问父类中具有 protected 访问权限的成员
- 包访问权限（没有关键字）：成员可以被**同一个包中的所有方法**访问，但是这个包之外的成员无法访问
- 私有的 - `private`



## 继承多态

将某一个抽象的类，改造成能够适用于不同特定需求的类。

```java
class Test extends father{
    private int birthday;
    private String sex;
    private int i;
	
    void info() {
        System.out.println("Tree is " + height + " feet tall");
    }
    // 重载
    @Overload // 该注解写与不写 JVM 都能自动识别方法重载。写上有助于 JVM 检查错误
    void info(String s) {
        System.out.println(s + ": Tree is " + height + " feet tall");
	}
    // 重写，在覆盖一个方法的时候，子类方法不能低于父类方法的可见性
    @Override
    void father_method() {
        System.out.println("ok");
    } 
    // this，只能在非静态方法内部使用
    // 也可在构造函数中调用另一个构造函数
   	Test increment() {
        i++;
        return this;
    }
    
}
// 向上转型
// 自动转换，一种多态
// 父类引用变量指向子类对象后，只能使用父类已声明的方法，但方法如果被重写会执行子类的方法，如果方法未被重写那么将执行父类的方法。
Animal animal = new Cat(...); 
// 向下转型
// 需要调用一些子类特有而父类没有的方法
Cat cat = (Cat) animal; 
```

**【Java 不支持多重继承】**

如果一个子类拥有多个父类的话，那么当多个父类中有重复的属性或者方法时，子类的调用结果就会含糊不清，也就是存在【二义性】

**【Java 不支持操作符重载】**

## 组合

在新类中创建现有类的对象，表示出来的是一种明确的**「整体-部分」**的关系

```java
public class Cat {
    // 组合
 	private Animal animal;
    // 使用构造函数初始化成员变量
    public Cat(Animal animal){
      this.animal = animal;
    }
    // 通过调用成员变量的固有方法使新类具有相同的功能
	public void breath(){
  		animal.breath();
    }
    // 通过调用成员变量的固有方法使新类具有相同的功能
    public void beat(){
 	 	animal.beat();
 	}
    // 为新类增加新的方法
 	public void run(){
  		System.out.println("I'm running");  
 	}
}
```



**慎用继承，优先使用组合**

使用继承就无法避免以下这两个问题：

1）打破了封装性，违反了 OOP 原则。迫使开发者去了解父类的实现细节，子类和父类耦合

2）父类更新后可能会导致一些不可知的错误

父类中可覆盖的方法调用了别的可覆盖的方法，这时候如果子类覆盖了其中的一些方法，就可能导致错误



**绑定**

将一个方法调用同一个方法主体关联起来的过程就称作绑定。

静态绑定：绑定发生在程序运行前，如 final 和 static 关键字

动态绑定：运行时根据对象的类型自动的进行绑定，动态绑定是多态的基础。



## 抽象

**抽象类**

```java
// 抽象类
public abstract class Person {
    // 抽象方法
 	public abstract String getDescription();
}
// 非抽象类
public class Student extends Person { 
    private String major; 
    public Student(String name, String major) { 
        super(name); 
        this.major = major; 
    } 
    @Override
    public String getDescription(){ // 实现父类抽象方法
     return "a student majoring in " + major; 
    } 
} 
Person p = new Student("Jack","Computer Science");// p 引用的是 Student 这样的【具体子类对象】
// 检查一个对象是否属于某个特定类或接口
if(x instanceof Student){
 ...
}
```

除了抽象方法之外，抽象类还可以包含具体数据和具体方法

abstract关键字只能用在抽象类中修饰方法，并且没有具体的实现。抽象方法的实现必须在派生类中使用override关键字来实现。



**接口**

用来提供公用的方法，规定子类的行为

提高了代码的可维护性和可扩展性，降低了代码的耦合度，不涉及任何具体的实现细节，比较安全

```java
// 接口
public interface Concept {
    void idea1();
    void idea2();
}

class Implementation implements Concept {
    @Override
    public void idea1() {
        System.out.println("idea1");
    }
    
    @Override
    public void idea2() {
        System.out.println("idea2");
    }
}
```

接口中的【属性】将被自动被设置为 `public static final` 类型；

接口中的【方法】将被自动被设置为 `public` 类型，并且不允许定义为 private 或者 protected。从 Java 9 开始，允许将方法定义为 private，这样就能定义某些复用的代码又不会把方法暴露出去。

一个类只能继承一个父类，但是一个类可以实现**【多个接口】**

在 Java 8 中，允许在接口中增加静态方法和默认方法。当冲突时：

1 )  **【超类优先】**。如果超类提供了一个具体方法，接口中的同名且有相同参数类型的默认方法会被忽略。

2 )  **【接口冲突】**。如果一个父类接口提供了一个默认方法，另一个父类接口也提供了一个同名而且参数类型相同的方法，子类必须覆盖这个方法来解决冲突。



**比较**

- 从设计层面上看，抽象类提供了一种 IS-A 关系，需要满足里式替换原则，即子类对象必须能够替换掉所有父类对象。而接口更像是一种 LIKE-A 关系，它只是提供一种方法实现契约，并不要求接口和实现接口的类具有 IS-A 关系。
- 从使用上来看，一个类可以实现多个接口，但是不能继承多个抽象类。
- 接口的字段只能是 static 和 final 类型的，而抽象类的字段没有这种限制。
- 接口的成员只能是 public 的，而抽象类的成员可以有多种访问权限。

**使用选择**

使用接口：

- 需要让不相关的类都实现一个方法，例如不相关的类都可以实现 Comparable 接口中的 compareTo() 方法；
- 需要使用多重继承。

使用抽象类：

- 需要在几个相关的类中共享代码。
- 需要能控制继承来的成员的访问权限，而不是都为 public。
- 需要继承非静态和非常量字段。

在很多情况下，接口优先于抽象类。因为接口没有抽象类严格的类层次结构要求，可以灵活地为一个类添加行为。并且从 Java 8 开始，接口也可以有默认的方法实现，使得修改接口的成本也变的很低。



# 反射机制

**优点**：

- **可扩展性**   ：应用程序可以利用全限定名创建可扩展对象的实例，来使用来自外部的用户自定义类。比较灵活，能够在运行时动态获取类的实例。
- **类浏览器和可视化开发环境**   ：一个类浏览器需要可以枚举类的成员。可视化开发环境（如 IDE）可以从利用反射中可用的类型信息中受益，以帮助程序员编写正确的代码。
- **调试器和测试工具**   ： 调试器需要能够检查一个类里的私有成员。测试工具可以利用反射来自动地调用类里定义的可被发现的 API 定义，以确保一组测试中有较高的代码覆盖率。

**缺点**：

尽管反射非常强大，但也不能滥用。如果一个功能可以不用反射完成，那么最好就不用。在我们使用反射技术时，下面几条内容应该牢记于心。

- **性能开销**   ：反射涉及了动态类型的解析，所以 **JVM 无法对这些代码进行优化**。因此，反射操作的效率要比那些非反射操作低得多。我们应该避免在经常被执行的代码或对性能要求很高的程序中使用反射。
- **安全限制**   ：使用反射技术要求程序**必须在一个没有安全限制的环境中运行**。如果一个程序必须在有安全限制的环境中运行，如 Applet，那么这就是个问题了。
- **内部暴露**   ：由于反射允许代码执行一些在正常情况下不被允许的操作（比如**访问私有的属性和方法**），所以使用反射可能会导致意料之外的副作用，这可能导致代码功能失调并破坏可移植性和封装性。反射代码破坏了抽象性，因此当平台发生改变的时候，代码的行为就有可能也随着变化。



 ## Class类

在程序运行期间，JVM 始终为所有的对象维护一个【运行时的类型标识】，这个信息跟踪着每个对象所属的类的完整结构信息，包括包名、类名、实现的接口、拥有的方法和字段等。

![图片](..\Resources\Class.png)

可以把 Class 类理解为【类的类型】，一个 Class 对象，称为类的类型对象，一个 Class 对象对应一个加载到 JVM 中的一个 .class 文件。

```java
import java.util.Date; // 先有类
public class Test {
    public static void main(String[] args) {
        Date date = new Date(); // 后有对象
        System.out.println(date);
    }
}
```

首先 JVM 会将你的代码编译成一个 `.class` 字节码文件，然后被类加载器（Class Loader）加载进 JVM 的内存中，**同时会创建一个 `Date` 类的 `Class` 对象存到堆中**（注意这个不是 new 出来的对象，而是类的类型对象）。JVM 在创建 `Date` 对象前，会先检查其类是否加载，寻找类对应的 `Class` 对象，若加载好，则为其分配内存，然后再进行初始化 `new Date()`。**每个类只有一个 `Class` 对象**，如果有第二条 `new Date()` 语句，JVM 不会再生成一个 `Date` 的 `Class` 对象。

反射的含义：**可以通过这个 `Class` 对象看到类的结构**



**获取 Class 类对象**

```java
// 1.类名.class，知道具体类
Class alunbarClass = TargetObject.class;
// 2.通过 Class.forName() 传入全类名获取
Class alunbarClass1 = Class.forName("com.xxx.TargetObject");
// 3.通过对象实例 instance.getClass() 获取
Class alunbarClass2 = date.getClass();
// 4.通过类加载器 xxxClassLoader.loadClass() 传入类路径获取
class clazz = ClassLoader.LoadClass("com.xxx.TargetObject");
```



## 反射获得类结构

**通过反射构造一个类的实例**

```java
Class<Object> clazz = (Class<Object>) Class.forName("fanshe.Student");
// 1.创建一个与 clazz 具有相同类类型的实例
// newInstance 方法【调用默认的构造函数（无参构造函数）】初始化新创建的对象。如果这个类没有默认的构造函数， 就会抛出一个异常
Date date2 = clazz.newInstance(); 

// 2.如果需要调用类的带参构造函数、私有构造函数等， 就需要采用 Constractor.newInstance()
// 1）获取所有"公有的"构造方法
Constructor[] conArray = clazz.getConstructors();
// 2）获取所有的构造方法（包括私有、受保护、默认、公有）
Constructor[] conArray = clazz.getDeclaredConstructors();
// 3）获取一个指定参数类型的"公有的"构造方法 
Constructor con = clazz.getConstructor(null); // 无参的构造方法类型是一个null
// 4）获取一个指定参数类型的"构造方法"，可以是私有的，或受保护、默认、公有
Constructor con = clazz.getDeclaredConstructor(int.class); 
con.setAccessible(true); // 为了调用 private 方法/域 我们需要取消安全检查

// 使用开源库 Objenesis
Objenesis objenesis = new ObjenesisStd(true);
Test test = objenesis.newInstance(Test.class);
test.show();
```



**通过反射获取成员属性**

```java
Class<Object> clazz = (Class<Object>) Class.forName("fanshe.Student");
// 获取所有公有的字段
Field[] fieldArray = clazz.getFields();
// 获取所有的字段（包括私有、受保护、默认的）
Field[] fieldArray = clazz.getDeclaredFields();
// 获取一个指定名称的公有的字段
Field f = clazz.getField("name");
// 获取一个指定名称的字段，可以是私有、受保护、默认的
Field f = clazz.getDeclaredField("phoneNum");

f.setAccessible(true); // 暴力反射，解除私有限定
f.set(obj, "刘德华"); // 为 Student 对象中的 name 属性赋值
```



**通过反射获取成员方法**

```java
Class<Object> clazz = (Class<Object>) Class.forName("fanshe.Student");
// 获取所有"公有方法"（包含父类的方法，当然也包含 Object 类）
Method[] methodArray  = clazz.getMethods();
// 获取所有的成员方法，包括私有的（不包括继承的）
Method[] methodArray  = clazz.getDeclaredMethods();
// 获取一个指定方法名和参数类型的成员方法：
Method m = clazz.getMethod("name");

m.invoke(obj, "小牛肉");
```





# 泛型

```java
public class Box<T> {
    // T stands for "Type"
    private T t;
    public void set(T t) { this.t = t; }
    public T get() { return t; }
}
```



# 注解











   