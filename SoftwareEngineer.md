@ 软件工程笔记
# 软件过程
最基本的软件工程活动：
 1. 软件规格说明
 2. 软件开发
 3. 软件确认
 4. 软件演化
## 软件过程模型
###  瀑布模型
1.需求分析
2.系统和软件设计
3.实现和单元测试
4.集成和系统测试
5.运行和维护

**要求早期承诺并且在实施变更时进行系统返工，很少使用**
**只适用于**：1.嵌入式系统（硬件不灵活）2.关键性系统（对安全性进行全面分析，文档必须完整）3.大型软件系统（完整规格说明以使不同子系统独立开发）
### 增量式开发
**先开发出一个初始版本，然后获得使用反馈并经过多个版本演化得到所需系统**
**优势**：1.降低变更需求成本
				2.更容易得到客户对已完成的开发工作的反馈意见
				3.更早获得价值
**劣势**：1.过程不可见，难以管理和掌握进度
				2.系统结构逐渐退化（定期重构）
## 软件过程活动
+ 系统规格说明
1.需求分析：得出系统需求
2.需求规格说明：转化为文档，包括用户需求和系统需求
3.需求确认：修改文档

+ 软件设计和实现
1.体系结构设计：总体结构，基本构件
2.数据库设计
3.接口设计：独立开发
4.构件选取

+ 软件确认
1.构件测试
2.系统测试
3.客户测试

+ 软件演化
# 敏捷软件开发
+ 需求被表达为**用户故事**，可用于规划系统迭代，其主要问题是完整性。
+ 及时重构避免修改时发生的自然结构退化
+ 即使是敏捷开发也需要系统需求文档

# 需求工程

大多数系统，开始前需要一个清晰可识别的需求（除了敏捷过程）

## 









## 编程过程

 1. 代码静态审查：代码有哪些错误，pylint
 2. 代码性能分析：分析模块耗时，profile

## 单元测试
保证单个质量即保证整体质量
 - 模块接口数据流测试
 - 局部数据结构
 - 边界条件
 - 独立路径，计算错误、判断错误、等
 - 出错处理
 pyunit，mock测试

 白盒测试：允许利用内部逻辑对程序所有逻辑路径测试