Spring 是分层的 Java SE/EE应用 full-stack 轻量级开源框架，<strong>以 IoC（Inverse Of Control： 控制反转）和 AOP（Aspect Oriented Programming：面向切面编程）为内核</strong>，提供了展现层 Spring MVC 和持久层 Spring JDBC 以及业务层事务管理等众多的企业级应用技术，还能整合开源世界众多著名的第三方框架和类库，逐渐成为使用最多的Java EE 企业应用开源框架。

<img src="..\Resources\spring framework runtime.png" alt="img" style="zoom:75%;" />

- **Spring Core：** 基础,可以说 Spring 其他所有的功能都需要依赖于该类库。主要提供 IoC 依赖注入功能。
- **Spring Aspects** ： 该模块为与 AspectJ 的集成提供支持。
- **Spring AOP** ：提供了面向切面的编程实现。
- **Spring JDBC** : Java 数据库连接。
- **Spring JMS** ：Java 消息服务。
- **Spring ORM** : 用于支持 Hibernate 等 ORM 工具。
- **Spring Web** : 为创建 Web 应用程序提供支持。
- **Spring Test** : 提供了对 JUnit 和 TestNG 测试的支持。

# IOC

IoC（Inverse of Control:控制反转）是一种**设计思想**，就是将原本在程序中手动创建对象的控制权，交由Spring框架来管理。 

IoC 容器是 Spring 用来实现 IoC 的载体，IoC 容器实际上就是个**Map（key，value）**,Map 中存放的是各种对象。

loC 容器就像是一个工厂一样，当我们需要创建一个对象的时候**，只需要配置好配置文件/注解即可**，完全不用考虑对象是如何被创建出来的。









# ApplicationContext实例化

```java
// 从类路径ClassPath中寻找
// 装载单个配置文件
ApplicationContext cxt = new ClassPathXmlApplicationContext("applicationContext.xml");
// 装载多个配置文件
String configs = {"bean1.xml","bean2.xml","bean3.xml"};
ApplicationContext cxt = new ClassPathXmlApplicationContext(configs);

// 从指定的文件系统路径中寻找
// 装载单个配置文件
ApplicationContext cxt = new FileSystemXmlApplicationContext("applicationContext.xml");
// 装载多个配置文件
String configs = {"c:/bean1.xml","c:/bean2.xml","c:/bean3.xml"};
ApplicationContext cxt = new FileSystemXmlApplicationContext(configs);

// 读取注解创建容器
AnnotationConfigApplicationContext
```



# Bean

## Bean 的创建

### Xml 配置

1. **使用默认无参构造函数**

   在 Spring 的配置文件中使用 bean 标签，配以 id 和 class 属性后，且没有其他属性和标签时。采用的就是默认构造函数创建 bean 对象；此时如果 bean（类） 中没有默认无参构造函数，将会创建失败

   ```xml
   <bean id = "accountService" class = "com.smallbeef.service.impl.AccountServiceImpl">
   ```

2. **使用简单工厂模式的方法创建**（使用某个类中的方法创建对象，并存入 Spring 容器）

   ```java
     /** 
      * 模拟一个工厂类
      * 该类可能是存在于jar包中的，我们无法通过修改源码的方式来提供默认构造函数
      * 此工厂创建对象，必须先有工厂实例对象，再调用方法  
      */ 
     public class InstanceFactory {   
         public IAccountService createAccountService(){   
             return new AccountServiceImpl();  
         }
     }
   ```

   ```xml
   <bean id = "InstanceFactory" class = "com.smallbeef.factory.InstanceFactory"></bean>
   
   <bean id="accountService"  
         factory-bean="InstanceFactory"     
         factory-method="createAccountService">
   </bean>
   ```

3. **使用静态工厂的方法创建对象**（使用某个类中的**静态方法**创建对象，并存入 Spring 容器）

    ```java
    /** 
     * 模拟一个静态工厂类
     * 该类可能是存在于jar包中的，我们无法通过修改源码的方式来提供默认构造函数
     */ 
    public class StaticFactory {   
        public static IAccountService createAccountService(){   
            return new AccountServiceImpl();  
        } 
    }
    ```

    ```xml
    <bean id="accountService"  
          class="com.smallbeef.factory.StaticFactory"     
          factory-method="createAccountService">
    </bean>
    ```

### 注解配置

以下注解的作用和在 XML 配置文件中编写一个 bean 标签实现的功能是一样的 , 用于把当前类对象存入 Spring 容器中

使用以下注解的时候，需要在 xml 文件中配置如下:（当然，其他的 bean 注册配置就不用写了，配合下面注解这一行就可以了）

```xml
 <!--告知Spirng在创建容器时要扫描的包，配置所需要的标签不是在beans的约束中，而是一个名称为context空间和约束中-->
<context:component-scan base-package="com.smallbeef"></context:component-scan>
```

- `@Component`：value属性——用于指定 bean 的 id 。当我们不写时，他的默认值是当前类名，且首字母小写。

- `@Controller`：一般用于【表现层】的注解。

- `@Service`：一般用于【业务层】的注解。

- `@Repository `：一般用于【持久层】的注解。 

上述四个注解可以随意互换, 作用相同,  都是用于用于把当前类对象存入 Spring 容器中, 只不过后面三个注解提供了更具体的语义化罢了

```java
// 没有写 value 默认值 'accountServiceImpl'
@Service 
public class AccountServiceImpl implements IAccountService {
 	// doSomething
}
```

也可使用配置类+ `@ComponentScan` 注解



## Bean 的获取

通过`ac.getBean`方法来从 Spring 容器中获取 Bean，传入的参数是 Bean 的 name 或者 id 属性

也可以直接通过 Class 去获取一个 Bean，但这种方式存在一个很大的弊端，如果存在多个实例（多个 Bean），这种方式就不可用。所以一般建议通过 name 或者 id 去获取 Bean 的实例

```java
// bean.xml
<bean id = "accountServiceImpl" class = "com.smallbeef.service.impl.AccountServiceImpl"></bean>
// .java
public class Client {
    public static void main(String[] args) {
        ApplicationContext ac = new ClassPathXmlApplicationContext("bean.xml");
        IAccountService aService = ac.getBean(accountServiceImpl);
        IAccountService aService = ac.getBean(IAccountService.class);
    }
}
```



## Bean 的作用范围

从 Spring 容器中多次获取同一个Bean，默认情况下，获取到的实际上是同一个实例，即默认是单例的。当然，我们可以手动配置

### Xml配置

```xml
<bean class = "com.smallbeef.dao.useDaoImpl" id = "userDao" scope = "prototype"/>
```

bean 标签的 `scope` 属性就是用来指定 bean 的作用范围的

- singleton：默认值，单例的（bean对象默认是单例模式）
- prototype：多例的
- request：作用于web应用的请求范围。WEB 项目中，Spring 创建一个 Bean 的对象，将对象存入到 request 域中
- session：作用于web应用的会话范围。WEB 项目中，Spring 创建一个 Bean 的对象，将对象存入到 session 域中
- global-session：作用于集群环境的会话范围。WEB 项目中，应用在 Portlet（集群） 环境。如果没有 Portlet 环境那么 global-session 相当于 session

### 注解配置

当然，除了使用 bean 标签在 xml 中进行配置，我们也可以在 Java 代码中使用注解 `@Scope` 来配置Bean的作用范围

```java
@Repository
@Scope("prototype")
public calss UserDao{
    public String hello(){
        return "hello";
    }
}
```



##  Bean 的生命周期 

- **单例对象**  `scope="singleton"`

  一个应用只有一个对象的实例。它的作用范围就是整个引用。

  生命周期：

  - 对象出生：当应用加载，创建容器时，对象就被创建了

  - 对象活着：只要容器在，对象一直活着。

  - 对象死亡：当应用卸载，销毁容器时，对象就被销毁了


  总结： **单例对象的生命周期和容器相同**

- **多例对象**  `scope="prototype"`

  每次访问对象时，都会重新创建对象实例。

  生命周期：

  - 对象出生：当使用对象时，才会创建新的对象实例

  - 对象活着：只要对象在使用中，就一直活着。

  - 对象死亡：**当对象长时间不用，且没有别的对象引用时，由 java 的垃圾回收器进行回收。**


### Xml 配置

bean 标签：

- `init-method`：指定类中的初始化方法名称。
- `destroy-method`：指定类中销毁方法名称。

```xml
<bean class = "com.smallbeef.dao.useDaoImpl" id = "userDao" scope = "prototype" init-method = "" destroy-method = ""/>
```

### 注解配置

- `@PreDestroy`

  作用：  用于指定销毁方法。

- `@PostConstruct `

  作用：  用于指定初始化方法。 















# Spring框架运行步骤

添加依赖包

```xml
<spring.version>5.0.4.RELEASE</spring.version>
<!--spring start -->
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-core</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-beans</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-context</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-context-support</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-aop</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-aspects</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-expression</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-tx</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-test</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-web</artifactId>
    <version>${spring.version}</version>
</dependency>
<!--spring end -->
```

运行容器后，监听context配置文件（此处为 applicationContext.xml）

```xml
web.xml
<!DOCTYPE web-app PUBLIC>
<web-app>
  <display-name>Archetype Created Web Application</display-name>
  <context-param>
    <param-name>contextConfigLocation</param-name>
    <param-value>classpath:applicationContext.xml</param-value>
  </context-param>
  <listener>
    <listener-class>org.springframework.web.context.ContextLoaderListener</listener-class>
  </listener>
</web-app>
```

注册，加载bean

【<context:component-scan base-package="com.szy"/>】：扫描 base-package 包下的所有java类，注册成Bean

```xml
/src/main/resources/applicationContext.xml
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xmlns:tx="http://www.springframework.org/schema/tx"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
           http://www.springframework.org/schema/beans/spring-beans-2.5.xsd
           http://www.springframework.org/schema/tx
           http://www.springframework.org/schema/tx/spring-tx.xsd
           http://www.springframework.org/schema/context
           http://www.springframework.org/schema/context/spring-context-2.5.xsd">

    <context:component-scan base-package="com.szy"/>
</beans>
```

使用 junit 运行测试类

@Service：服务层

@Test：告诉junit是单元测试方法

```java
SpringTest.java
@Service
public class SpringTest {
    @Test
    public void testSpring() {
        ApplicationContext applicationContext =
                new ClassPathXmlApplicationContext("applicationContext.xml");
        // 默认的Bean名字是首字母小写的类名
        SpringTest springTest = (SpringTest) applicationContext.getBean("springTest");
        springTest.sayHello();
    }
    public void sayHello() {
        System.out.println("hello szy");
    }

}
```







# SpringMVC运行步骤

添加依赖包

```xml
<spring.version>5.0.4.RELEASE</spring.version>
<javax.servlet.version>4.0.0</javax.servlet.version>
<jstl.version>1.2</jstl.version>
<!--springmvc start -->
<dependency>
    <groupId>jstl</groupId>
    <artifactId>jstl</artifactId>
    <version>${jstl.version}</version>
</dependency>
<dependency>
    <groupId>javax.servlet</groupId>
    <artifactId>javax.servlet-api</artifactId>
    <version>${javax.servlet.version}</version>
</dependency>
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-webmvc</artifactId>
    <version>${spring.version}</version>
</dependency>
<!--springmvc end -->
```

运行容器，配置servlet，加载SpringMVC的配置文件，监听context配置文件

【DispatcherServlet 类】：前置控制器，用于拦截匹配请求，分发到目标 Controller 处理

【\<init-param\>】：当前\<servlet\>中的局部变量，声明了配置文件目录

【\<load-on-startup\>】：值大于等于0时，表示容器在应用启动时加载并初始化此 servlet；小于0或未指定时，表示容器在该 servlet 被选择时才加载

【\<servlet-mapping\>】：声明了与该 servlet 相应的匹配规则，每个【\<url-pattern\>】标签代表一个匹配规则

```xml
web.xml
<!DOCTYPE web-app PUBLIC
 "-//Sun Microsystems, Inc.//DTD Web Application 2.3//EN"
 "http://java.sun.com/dtd/web-app_2_3.dtd" >
<web-app>
  <display-name>Archetype Created Web Application</display-name>

  <context-param>
    <param-name>contextConfigLocation</param-name>
    <param-value>classpath:applicationContext.xml</param-value>
  </context-param>

  <!--配置DispatcherServlet -->
  <servlet>
    <servlet-name>spring-dispatcher</servlet-name>
    <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
    <!-- 配置SpringMVC需要加载的配置文件 spring-mvc.xml -->
    <init-param>
      <param-name>contextConfigLocation</param-name>
      <param-value>classpath:spring-mvc.xml</param-value>
    </init-param>
    <load-on-startup>1</load-on-startup>
  </servlet>
  <servlet-mapping>
    <servlet-name>spring-dispatcher</servlet-name>
    <!--默认匹配所有的请求 -->
    <url-pattern>/</url-pattern>
  </servlet-mapping>

  <listener>
    <listener-class>org.springframework.web.context.ContextLoaderListener</listener-class>
  </listener>
</web-app>
```

配置SpringMVC

【\<mvc:annotation-driven/\>】：自动注册【RequestMappingHandlerAdapter】和【RequestMappingHandlerMapping】，是 SpringMVC 为 @Controller 分发请求所必需的，并提供了多种支持

【InternalResourceViewResolver】：最常用的视图解析器，当 @Controller 返回 ”Hello“ 时，解析器会自动添加前缀和后缀 ”/WEB-INF/views/hello.jsp“

```xml
spring-mvc.xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xmlns:mvc="http://www.springframework.org/schema/mvc"
       xmlns:aop="http://www.springframework.org/schema/aop"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
        http://www.springframework.org/schema/beans/spring-beans.xsd
        http://www.springframework.org/schema/context
        http://www.springframework.org/schema/context/spring-context.xsd
        http://www.springframework.org/schema/mvc
        http://www.springframework.org/schema/mvc/spring-mvc.xsd
        http://www.springframework.org/schema/aop
        http://www.springframework.org/schema/aop/spring-aop.xsd">

    <!-- 扫描controller(后端控制器),并且扫描其中的注解-->
    <context:component-scan base-package="com.szy.controller"/>
    <!--设置配置方案 -->
    <mvc:annotation-driven/>
    <!--配置JSP　显示ViewResolver(视图解析器)-->
    <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
        <property name="viewClass" value="org.springframework.web.servlet.view.JstlView"/>
        <property name="prefix" value="/WEB-INF/views/"/>
        <property name="suffix" value=".jsp"/>
    </bean>
</beans>
```

将@Controller实例化一个单例对象 SzyTestController

```java
szyTestController.java
package com.szy.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping(value = "/test")
public class SzyTestController {
    @GetMapping("/sayHello")
    public String sayHello() {
        return "hello";
    }
}
```



# MyBatis运行步骤

添加依赖包

【mysql-connector-java】：MySQL的 JDBC 驱动包

【druid】：阿里巴巴一个数据库连接池实现，可以很好的监控DB连接池和SQL的执行情况

【mybatis-spring】：将 mybatis 代码无缝整合到 Spring 中

```xml
<mybatis.version>3.4.6</mybatis.version>
<mysql.connector.java.version>8.0.9-rc</mysql.connector.java.version>
<druid.version>1.1.9</druid.version>
<mybatis.spring.version>1.3.2</mybatis.spring.version>
<!--mybatis start -->
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <version>${mysql.connector.java.version}</version>
    <scope>runtime</scope>
</dependency>

<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>druid</artifactId>
    <version>${druid.version}</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-jdbc</artifactId>
    <version>${spring.version}</version>
</dependency>

<dependency>
    <groupId>org.mybatis</groupId>
    <artifactId>mybatis</artifactId>
    <version>${mybatis.version}</version>
</dependency>

<dependency>
    <groupId>org.mybatis</groupId>
    <artifactId>mybatis-spring</artifactId>
    <version>${mybatis.spring.version}</version>
</dependency>
<!--mybatis end -->
```

添加配置 /src/main/resources/jdbc.properties 

```properties
jdbc.driverClassName=com.mysql.jdbc.Driver
jdbc.url=jdbc:mysql://127.0.0.1:3306/springmvc-mybatis-book?serverTimezone=GMT
jdbc.username=root
jdbc.password=martin123
```

 配置Bean

【context:property-placeholder】：外在化参数配置。location 表示属性文件位置，多个属性文件之间逗号分隔；ignore-unresolvable 表示是否忽略解析不到的属性，如果不忽略，找不到将抛出异常

【DruidDataSource】：阿里巴巴Druid数据源，该数据源会读取 jdbc.properties 配置文件的数据库连接信息和驱动

【SqlSessionFactoryBean】：在 MyBatis-Spring 中，使用 SqlSessionFactoryBean 创建 Session 工厂；在基本的 MyBatis 中，使用 SqlSessionFactoryBuilder

【MapperScannerConfigurer】：查找类路径下的映射器并自动将它们创建成 MapperFactoryBean

```xml
/src/main/resources/applicationContext.xml
<!--1、配置数据库相关参数-->
<context:property-placeholder location="classpath:jdbc.properties" ignore-unresolvable="true"/>

<!--2.数据源 druid -->
<bean id="dataSource" class="com.alibaba.druid.pool.DruidDataSource" init-method="init" destroy-method="close">
    <property name="driverClassName" value="${jdbc.driverClassName}"/>
    <property name="url" value="${jdbc.url}"/>
    <property name="username" value="${jdbc.username}"/>
    <property name="password" value="${jdbc.password}"/>
</bean>

<!--3、配置SqlSessionFactory对象-->
<bean id="sqlSessionFactory" class="org.mybatis.spring.SqlSessionFactoryBean">
    <!--注入数据库连接池-->
    <property name="dataSource" ref="dataSource"/>
    <!--扫描sql配置文件:mapper需要的xml文件-->
    <property name="mapperLocations" value="classpath:mapper/*.xml"/>
    <!-- mybatis配置文件的位置 -->
    <property name="configLocation" value="classpath:mybatis-config.xml"></property>
    <!-- 配置分页插件 -->
    <property name="plugins">
        <array>
            <bean class="com.github.pagehelper.PageInterceptor">
                <property name="properties">
                    <value>
                        helperDialect=mysql
                        reasonable=true
                    </value>
                </property>
            </bean>
        </array>
    </property>

</bean>

<bean id="sqlSession" class="org.mybatis.spring.SqlSessionTemplate">
    <constructor-arg index="0" ref="sqlSessionFactory"/>
</bean>

<!-- 扫描basePackage下所有以@MyBatisDao注解的接口 -->
<bean id="mapperScannerConfigurer" class="org.mybatis.spring.mapper.MapperScannerConfigurer">
    <property name="sqlSessionFactoryBeanName" value="sqlSessionFactory"/>
    <property name="basePackage" value="com.ay.dao"/>

</bean>
```

创建数据库表对应的【实体类对象 】

```java
/src/main/java/com.szy.model/SzyUser.java
public class SzyUser implements Serializable {
    private Integer id;
    private String name;
    private String password;
    // 省略get，set方法
}
```

创建对应的DAO【对象接口】 SzyUserDao

```java
/src/main/java/com.szy.dao/SzyUserDao.java
package com.szy.dao;

import com.szy.model.SzyUser;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface SzyUserDao {
    List<SzyUser> findAll();
}
```

创建对应的【服务层接口】 SzyUserService

```java
/src/main/java/com.szy.services/SzyUserService.java
package com.szy.service;

import com.szy.model.SzyUser;
import java.util.List;

public interface SzyUserService {
    List<SzyUser> findAll();
}
```











