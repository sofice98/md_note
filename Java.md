JavaSE：标准版（桌面程序，控制台）
JavaME：嵌入式开发（手机）
JavaEE：E企业级开发（web，服务器）



JDK(Java Development Kit)：整个java开发工具
JRE(Java Runtime Environment)：运行时环境
JVM(Java Virtual Machine)：java虚拟机

Java 程序其实是运行在JVM (Java虚拟机) 上的，使用 Java 编译器编译 Java 程序时，生成的是与平台无关的字节码，**这些字节码只面向 JVM**。不同平台的 JVM 都是不同的，但它们都提供了相同的接口，这也正是 Java 跨平台的原因。

普通用户只需要安装 JRE 来运行 Java 程序。而程序开发者必须安装 JDK 来编译、调试程序

![image-20210304101524025](Java\jdk.png)

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



![图片](Java\类型合法转换.jpg)

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



**【基本类型比较可用 `==`，对象比较最好用`n1.equals(n2)`，该方法用于判断两个对象是否具有相同的引用（地址）】**

1）：类没有覆盖 `equals()` 方法。则通过 `equals()` 比较该类的两个对象时，等价于通过 `==` 比较这两个对象（比较的是地址）。

2）：类覆盖了 `equals()` 方法。一般来说，我们都覆盖 `equals()` 方法来判断两个对象的内容是否相等，比如 `String` 类。



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

**任何一个 Java 对象都可以转换成字符串**

JVM 为了提高性能和减少内存开销，在实例化字符串常量的时候进行了一些**优化**：为字符串开辟了一个**【字符串常量池 String Pool】**，可以理解为缓存区。创建字符串常量时，首先检查字符串常量池中是否存在该字符串，若字符串常量池中存在该字符串，则直接返回该引用实例，无需重新实例化；若不存在，则实例化该字符串并放入池中。

JDK 1.7 之前，字符串常量池存在于【常量存储（Constant storage）】中；JDK 1.7 之后，字符串常量池存在于【堆内存（Heap）】中

```java
String str1 = "hello"; // 分配到常量池中
String str2 = new String(“hello”); // 先在String Pool 中开辟地址空间创建一个字符串对象，指向这个 "hello" 字符串字面量，然后在堆中创建一个字符串对象，使引用指向堆中的对象
String str3 = str2.intern(); // 如果 String Pool 中已经存在一个字符串和该字符串的值相等，那么就会返回 String Pool 中字符串的引用；否则，就会在 String Pool 中添加一个新的字符串，并返回这个新字符串的引用
```



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

静态机制允许我们无需创建对象就可以直接通过类的引用来调用该方法

使用类名直接引用静态变量或方法是首选方案，因为它强调了静态属性

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





## 垃圾回收

Java 有一个**垃圾回收器**，用来监视 `new`创建的所有对象，并辨别那些不会被再引用的对象，然后释放这些对象的内存空间。

**finalize()**：当使用了内存之外的其他资源时使用，确保释放实例占用的全部资源。当垃圾收集器认为没有指向对象实例的引用时，会在销毁该对象之前调用 `finalize()` 方法。不过，在实际应用中，不要依赖于使用 `finalize`方法回收任何短缺的资源， 这是因为 **Java 并不保证定时为对象实例调用该方法，甚至不保证方法会被调用**，所以该方法不应该用于正常内存处理。





# 异常处理

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

代码块的调用顺序为：

1. 静态初始化块：使用 static 定义代码块，只有当类装载到系统时执行一次，之后不再执行。在静态初始化块中仅能初始化 static 修饰的数据成员
2. 非静态初始化块
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

单继承：

**在 Java 中，子类只能继承一个父类**。如果一个子类拥有多个父类的话，那么当多个父类中有重复的属性或者方法时，子类的调用结果就会含糊不清，也就是存在**「二义性」**



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

接口中的【属性】将被自动被设置为 `public static final` 类型；接口中的【方法】将被自动被设置为 `public` 类型

一个类只能继承一个父类，但是一个类**可以实现多个接口**

在 Java 8 中，允许在接口中增加静态方法和默认方法。当冲突时：

1 )  **「超类优先」**。如果超类提供了一个具体方法，接口中的同名且有相同参数类型的默认方法会被忽略。

2 )  **「接口冲突」**。如果一个父类接口提供了一个默认方法，另一个父类接口也提供了一个同名而且参数类型相同的方法，子类必须覆盖这个方法来解决冲突。





# 反射机制

**「优点」**：比较灵活，能够在运行时动态获取类的实例。

**「缺点」**：1）性能瓶颈：反射相当于一系列解释操作，通知 JVM 要做的事情，性能比直接的 Java 代码要慢很多。
				   2）安全问题：反射机制破坏了封装性，因为通过反射可以获取并调用类的私有方法和字段。

 ## Class类

在程序运行期间，JVM 始终为所有的对象维护一个【运行时的类型标识】，这个信息跟踪着每个对象所属的类的完整结构信息，包括包名、类名、实现的接口、拥有的方法和字段等。

![图片](Java\Class.png)

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





















