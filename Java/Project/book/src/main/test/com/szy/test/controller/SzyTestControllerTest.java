package com.szy.test.controller;

import com.szy.controller.SzyTestController;
import org.junit.Test;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.stereotype.Service;

@Service
public class SzyTestControllerTest {
    @Test
    public void testSpring() {
        ApplicationContext cxt =
                new ClassPathXmlApplicationContext("applicationContext.xml");
        SzyTestController szyTestController = (SzyTestController) cxt.getBean("szyTestController");
        System.out.println(szyTestController.sayHello());

    }
}
