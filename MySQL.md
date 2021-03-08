# 连接C#

## 配置

NuGet中添加MySql.Data

```c#
using MySql.Data.MySqlClient;
```

## 连接

```c#
string s = "server=localhost;database=test1;user=root;password=1234;pooling=true;charset=utf8;";
MySqlConnection connection = new MySqlConnection(s)；
    
try{
    connection.Open();
    Console.WriteLine("已经建立连接");
    //执行命令  
    connection.Close();
}
catch (MySqlException ex){
    Console.WriteLine(ex.Message);
}
```

【server】=服务器IP地址;
【database】=数据库名称;
【user】=数据库用户名;
【password】=数据库密码;
【pooling】=是否放入连接池;
【charset】=编码方式;

## 执行命令

https://blog.csdn.net/yellowman2/article/details/88724686?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&dist_request_id=1328576.13056.16146765333312607&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control

```c#
string sql = "select count(*) from tb_User where ID=123"；
MySqlCommand command = new MySqlCommand(sql, connection);
MySqlDataReader res = command.ExecuteReader();
reader.Read();//读取DataReader对象单行数据
reader.GetInt32(0); reader.GetString(1);
reader.GetInt32("userid");reader.GetString("username");//获取单行字段数据

//获取数据
DataSet GetDataSet(string sql)
{
    // 创建数据适配器,传入SQL语句和链接对象
    MySqlDataAdapter myAd = new MySqlDataAdapter(sql, connection);
    // 创建数据存放集合;
    DataSet data = new DataSet();
    // 通过数据适配器读取数据并导入到数据集合;
    myAd.Fill(data);
    //返回数据集
    return data;                                              
}
if (data.Tables[0].Rows.Count > 0)                  //判断是否存在
{
    // 循环表
    for (int i = 0; i < data.Tables.Count;i++ )
    {
        // 循环该表的行
        for (int j = 0; j < data.Tables[i].Rows.Count;j++ )
        {
            // 循环该表的列
            for (int k = 0; k < data.Tables[i].Columns.Count;k++ )
            {
                // 打印数据
                Console.Write(data.Tables[i].Rows[j][k]+"\t");
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }
}
```
ExcuteScalar：执行单行查询，返回查询结果的首行数据
ExcuteReader：执行多行查询，返回DataReader对象
ExcuteNonQuery：执行【insert（增）】、【updata（改）】、【delete（删）】语句