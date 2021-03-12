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
```













# Spring框架运行步骤

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
applicationContext.xml
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

运行容器，配置servlet，加载SpringMVC的配置文件，监听context配置文件

【DispatcherServlet 类】：前置控制器，用于拦截匹配请求，分发到目标 Controller 处理

【\<init-param\>】：当前\<servlet\>中的局部变量

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





