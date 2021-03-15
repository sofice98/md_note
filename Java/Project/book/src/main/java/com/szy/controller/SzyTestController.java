package com.szy.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping(value = "/test")
public class SzyTestController {
    @GetMapping("/szy")
    public String sayHello() {
        return "hello";
    }
}
