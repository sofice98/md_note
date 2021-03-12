# 构建工具的作用

构建一个项目通常包含了依赖管理、测试、编译、打包、发布等流程，构建工具可以自动化进行这些操作，从而为我们减少这些繁琐的工作。



**Java 主流构建工具**

Ant 具有编译、测试和打包功能，其后出现的 Maven 在 Ant 的功能基础上又新增了依赖管理功能，而最新的 Gradle 又在 Maven 的功能基础上新增了对 Groovy 语言的支持。

<img src="..\Resources\java构建工具.png" alt="img" style="zoom:67%;" />



# Maven

**Maven 主要用处一：相同的项目结构**

使用Maven管理的Java 项目都有着相同的项目结构

1. 有一个pom.xml 用于维护当前项目都用了哪些jar包
2. 所有的java代码都放在 src/main/java 下面
3. 所有的测试代码都放在src/test/java 下面

**Maven 主要用处二：统一维护jar包**

maven风格的项目，首先把所有的jar包都放在【仓库】里，然后哪个项目需要用到这个jar包，只需要给出jar包的名称和版本号就行了。 这样jar包就实现了共享



配置：https://how2j.cn/k/idea/idea-maven-config/1353.html

## 仓库

Maven 仓库能帮助我们管理构件（主要是JAR），它就是放置所有JAR文件（WAR，ZIP，POM等等）的地方。

Maven 仓库有三种类型：

- **本地（local）**

  运行 Maven 的时候，Maven 所需要的任何构件都是直接从本地仓库获取的。如果本地仓库没有，它会首先尝试从远程仓库下载构件至本地仓库，然后再使用本地仓库的构件。

- **中央（central）**

  由 Maven 社区提供的仓库，其中包含了大量常用的库。要浏览中央仓库的内容，maven 社区提供了一个 URL：https://search.maven.org/#browse

- **远程（remote）**

  如果 Maven 在中央仓库中也找不到依赖的文件，它会停止构建过程并输出错误信息到控制台。为避免这种情况，Maven 提供了远程仓库的概念，它是开发人员自己定制仓库，包含了所需要的代码库或者其他工程中用到的 jar 文件。



## POM

POM( Project Object Model，项目对象模型 ) 是 Maven 工程的基本工作单元，是一个XML文件，保存在项目根目录的 pom.xml 文件中，包含了项目的基本信息，用于描述项目如何构建，声明项目依赖，等等。

执行任务或目标时，Maven 会在当前目录中查找 POM。它读取 POM，获取所需的配置信息，然后执行目标。

```xml
<dependency>
    <groupId>junit</groupId>
    <artifactId>junit</artifactId>
    <version>4.12</version>
    <scope>test</scope>
</dependency>
```

[groupId, artifactId, version, packaging,  classifier] 称为一个项目的坐标，其中 groupId、artifactId、version 必须定义，packaging  可选（默认为 Jar），classifier 不能直接定义的，需要结合插件使用。

| 节点         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| project      | 工程的根标签                                                 |
| modelVersion | 模型版本，需要设置为 4.0                                     |
| groupId      | 工程组的标识。它在一个组织或者项目中通常是唯一的。例如，一个银行组织 com.companyname.project-group 拥有所有的和银行相关的项目 |
| artifactId   | 工程的标识。它通常是工程的名称。例如，消费者银行。groupId 和 artifactId 一起定义了 artifact 在仓库中的位置 |
| version      | 工程的版本号。在 artifact 的仓库中，它用来区分不同的版本。例如： `com.company.bank:consumer-banking:1.0 com.company.bank:consumer-banking:1.1` |
| packaging    | 项目打包方式。默认 jar                                       |











## 依赖原则

1. **依赖路径最短优先原则**

    ```html
    A -> B -> C -> X(1.0)
    A -> D -> X(2.0)
    ```

	由于 X(2.0) 路径最短，所以使用 X(2.0)。

2. **声明顺序优先原则**

    ```html
    A -> B -> X(1.0)
    A -> C -> X(2.0)
    ```

	在 POM 中最先声明的优先，上面的两个依赖如果先声明 B，那么最后使用 X(1.0)。

3. **覆写优先原则**

   子 POM 内声明的依赖优先于父 POM 中声明的依赖。





















