# # 简易的实现百度搜索功能
# # 作者：尚墨
# # 创建时间：2021年9月26日
# # 更新时间：2021年9月26日

# from time import sleep
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys

# # 实例化浏览器对象
# driver = webdriver.Edge()
# # 打开网址url
# driver.get('https://www.baidu.com/')
# # 搜索
# search_box = driver.find_element(By.ID, 'kw')
# search_box.send_keys('世界上最大的淡水湖')

# search_box.send_keys(Keys.RETURN)
# # 观察效果
# sleep(3)
# # 关闭网页
# driver.quit()

"""
    自动化测试——selenium"的使用
    一、selenium的简介
    Selenium是一个自动化测试工具，主要用于Web应用程序的测试。Selenium测试直接运行在浏览器中，就像真正的用户在操作一样。支持多种浏览器，包括Chrome、Firefox、Safari等主流界面浏览器。
    Selenium支持多种操作系统，包括Windows、Linux、Mac等。
    Selenium支持多种编程语言，包括Java、Python、C#、JavaScript等。
    Selenium支持多种测试框架，包括JUnit、TestNG等。
    Selenium支持多种测试模式，包括本地模式、远程模式。
    二、selenium的安装
    1.安装selenium
    pip install selenium
    2.下载浏览器驱动
    下载地址：https://sites.google.com/a/chromium.org/chromedriver/downloads
    下载后，将驱动程序放到Python的安装目录下即可。
    三、selenium的使用
    1.导入selenium模块
    from selenium import webdriver
    2.实例化浏览器对象
    driver = webdriver.Chrome()
    3.打开网址
    driver.get('https://www.baidu.com/')
    4.查找元素
    driver.find_element_by_id('kw')
    5.操作元素
    element.send_keys('Python')
    6.关闭网页
    driver.quit()
    四、selenium的常用方法
    1.查找元素
    find_element_by_id()：通过id查找元素
    find_element_by_name()：通过name查找元素
    find_element_by_xpath()：通过xpath查找元素
    find_element_by_link_text()：通过链接文本查找元素
    find_element_by_partial_link_text()：通过部分链接文本查找元素
    find_element_by_tag_name()：通过标签名查找元素
    find_element_by_class_name()：通过类名
    find_element_by_css_selector()：通过css选择器
    2.操作元素
    send_keys()：输入文本
    click()：点击元素
    clear()：清空文本
    3.获取元素属性
    get_attribute()：获取元素属性
    4.获取元素文本
    text：获取元素文本
    5.等待
    sleep()：等待时间
    implicitly_wait()：隐式等待
    6.关闭网页
    quit()：关闭网页

"""



# 八、实例：实现百度搜索功能
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# 实例化浏览器对象
driver = webdriver.Edge()
# 打开网址url
driver.get('https://www.baidu.com/')
# 搜索
search_box = driver.find_element(By.ID, 'kw')
search_box.click()
# search_box.send_keys(Keys.RETURN)
search_box.send_keys('世界上最大的淡水湖')

search_box.submit()

# 观察效果
sleep(3)

# 关闭网页
driver.quit()



# from selenium import webdriver  
# from selenium.webdriver.common.by import By  
# from selenium.webdriver.chrome.service import Service  

# # 创建 ChromeDriver 服务  
# service = Service('C:\Program Files\chrome-win64')  # 确保这里是 ChromeDriver 的正确路径  
# driver = webdriver.Chrome(service=service)  

# driver.get('https://www.example.com')  # 替换为你需要访问的网站  

# # 使用新的查找方法
# search_box = driver.find_element(By.ID, 'kw')  
# search_box.send_keys('易烊千玺')  

# # 记得完成后关闭浏览器  
# driver.quit()

