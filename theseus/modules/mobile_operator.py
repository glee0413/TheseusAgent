import logging
import time
from xxlimited_35 import error

from anyio.abc import value
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
import os
from collections import namedtuple

from appium.webdriver.extensions.android.nativekey import AndroidKey
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException,InvalidSelectorException
import appium.webdriver.extensions.android.nativekey as nativekey

from loguru import logger


capabilities = dict(
    platformName='Android',
    automationName='uiautomator2',
    deviceName='Android',
    # appPackage='com.android.settings',
    appActivity='com.android.ui.SplashActivity',
    unlockType = 'password', # pattern手势
    unlockKey = '123698'

    # language='zh',
    # locale='CN'

)


# AppInfo = namedtuple('AppInfo', ['name', 'object', 'desc', 'type'])


class ReturnValue:
    def __init__(self,flag,obj = None,message = 'OK'):
        self.flag = flag
        self.obj = obj
        self.llm_message = message

class AppInfo:
    def __init__(self, name, obj, desc, screen_id):
        self.name = name
        self.android_app = obj
        self.desc = desc
        self.screen_id = type

# 按照页面建立一颗树
class AppBank:
    def __init__(self):
        self.bank = {} #key: app name; value : app ref

    def insert_app(self,tag_name : str,screen_idx,element):
        if not tag_name:
            logger.warning(f'tag_name is <{tag_name}>')
            return  ReturnValue(flag=False,obj=None,message='app name is none')

        appname = tag_name.split()[0]
        print(f'appname {appname}')
        app_unread = tag_name.split(' ')[1]

        app_info = AppInfo(appname, element, desc = app_unread, screen_id = screen_idx)
        self.bank[appname] = app_info
        logger.info(f'Insert app {appname}')

        return ReturnValue(flag=True,obj=app_info)

    def get_app(self, app_name):
        if app_name in self.bank:
            logger.info(f'find {app_name}')
            return ReturnValue(flag=True,obj=self.bank[app_name])

        logger.info(f'cant find {app_name}')
        return ReturnValue(flag=False,obj=None)

class AndroidElement:
    def __init__(self,id,xpath,ui2, text = ''):
        self.id = id
        self.xpath = xpath
        self.text = text
        self.ui2 = ui2

class MobileOperator():
    def __init__(self):
        logger.info(f'MobileOperator init!')
        appium_server_url = 'http://localhost:4723'
        # cost 4.3s
        self.driver = webdriver.Remote(appium_server_url, options=UiAutomator2Options().load_capabilities(capabilities))
        logger.info(f'webdriver init finish')

        # self.goto_screen()
        self.app_bank = AppBank()
        self.screen_views = {}
        self.screen_views_name = ['智能助理']
        self.screen_selector = {}
        self.all_app = {}
        self.width = self.driver.get_window_size()['width']
        self.height = self.driver.get_window_size()['height']

        self.workspace = AndroidElement(
            id = 'com.miui.home:id/workspace',
            xpath='//com.miui.home.launcher.ScreenView[@resource-id="com.miui.home:id/workspace"]',
            ui2 = 'new UiSelector().resourceId("com.miui.home:id/workspace"'
        )
        self.hotseat = AndroidElement(
            id = 'com.miui.home:id/hotseat',
            xpath = '//com.miui.home.launcher.ScreenView[@resource-id="com.miui.home:id/hotseat"]',
            ui2 = 'new UiSelector().resourceId("com.miui.home:id/hotseat")'
        )
        self.screen_view_frame = AndroidElement(
            id = '',
            xpath = '//com.miui.home.launcher.ScreenView[@resource-id="com.miui.home:id/workspace"]/android.widget.FrameLayout[1]/com.miui.home.launcher.ScreenView',
            ui2= 'new UiSelector().className("com.miui.home.launcher.ScreenView").instance(1)'
        )

    def get_mobile_info(self):
        logging.info(f"get_mobile_info start")
        self.get_screen_view_selector()
        logging.info(f"get_screen_view over")
        self.get_all_apps()
        logging.info(f"get_all_apps over")

    def function_key(self,key = AndroidKey.HOME):
        self.driver.press_keycode(int(key))
        return ReturnValue(flag=True)

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int):
        logger.info(f'swipe : ({start_x},{start_y}),({end_x},{end_y})')
        self.driver.swipe(start_x, start_y,
                          end_x, end_y, 500)

    def get_elements(self, value="//*[@clickable='true']"):
        elements = self.driver.find_elements(by=AppiumBy.XPATH,value = value)
        return elements

    def get_hotseat(self):
        ret = self.function_key(AndroidKey.HOME)
        return ReturnValue(flag=True)

    def get_screen_view_selector(self):
        ret = self.function_key(AndroidKey.HOME)
        if not ret.flag:
            return ret

        try:
            screen_view_xpath = f'{self.workspace.xpath}/*/com.miui.home.launcher.ScreenView/*'
            logger.info(f'screen_view_xpath : {screen_view_xpath}')
            self.screen_selector = self.driver.find_elements(
                by = AppiumBy.XPATH,
                value = screen_view_xpath)

            screen_names = [ screen.tag_name for screen in self.screen_selector ]
            logger.info(f'Get views {len(self.screen_selector)} names {screen_names}')
        except NoSuchElementException as e:
            logger.warning(f'Get screen_view_xpath child failed')
            return ReturnValue(flag=False, message="Get view frame child failed")

        return ReturnValue(flag = True,obj = self.screen_views)

    def goto_screen_view(self, screen_id: int = 0):
        if screen_id >= len(self.screen_selector):
            return
        try:
            self.screen_selector[screen_id].click()
        except StaleElementReferenceException as e:
            self.driver.press_keycode(3)
            self.screen_selector[screen_id].click()

    def get_all_apps(self):
        for idx in range(1,len(self.screen_selector)):
            self.goto_screen_view(idx)
            app_pattern = f"{self.workspace.xpath}/*/android.view.ViewGroup/*[@clickable='true']"
            logger.info(f'app_pattern {app_pattern}')
            try:
                apps = self.driver.find_elements(by=AppiumBy.XPATH,value = app_pattern)
            except NoSuchElementException as e:
                logger.warning(f'Error: {str(e)}')
                return ReturnValue(flag=False)
            except InvalidSelectorException as e:
                logger.warning(f'Error: {str(e)}')
                return ReturnValue(flag=False)

            for app in apps:
                app_name = app.tag_name
                logger.info(f'app {app_name}')
                self.app_bank.insert_app(tag_name=app_name,screen_idx=idx,element=app)

    def tap_app(self,app_name):
        app = self.app_bank.get_app(app_name)
        if not app.flag:
            return app

        appinfo:AppInfo = app.obj
        self.function_key()
        self.goto_screen_view(app.obj.screen_id)
        appinfo.android_app.click()

        return ReturnValue(flag=True,obj=None)

    def print_app(self):
        for app in self.all_app:
            print(f'appname: {app.name} - apptype:{app.type}')
        for idx,screen in enumerate(self.screen_views):
            print(f"######################## {idx} ################################")
            screen_apps = self.app_bank.get_screen_apps(idx)
            for app in screen_apps:
                print(f'app name: {app.name} app type -- {app.type}')

    # def init_desktop(self,n):


def test_home_page(mobile_operator):
    elements = mobile_operator.get_elements()
    named_ele = [ele for ele in elements if ele.text is not None and ele.text.strip()!= ""]

    for ele in elements:
        print(ele.get_attribute('class'),'<text:',ele.text,'> <tag name',ele.tag_name, '> <clickable:',ele.get_attribute('clickable'),'>')

def test_screen_view(mobile_operator):
    # elements = mobile_operator.get_screen_view()
    # print(f'get screen count {len(elements)}')
    # for ele in elements:
    #     print(f'type: {type(ele)}')
    #     print(ele.get_attribute('class'),'<text:',ele.text,'> <tag name',ele.tag_name, '> <selected:',ele.get_attribute('selected'),'>')
    return None

def test_all_app(mobile_operator:MobileOperator):
    mobile_operator.goto_screen()


    # mobile_operator.print_app()
def test_screen_swipe(mobile_operator:MobileOperator):
    mobile_operator.goto_screen('第3屏')
    time.sleep(3)
    mobile_operator.goto_screen('第1屏')
    time.sleep(3)
    mobile_operator.goto_screen('第4屏')

def test_get_screen_views(mobile_operator:MobileOperator):
    mobile_operator.get_screen_view_selector()
    mobile_operator.get_all_apps()
    logging.info(f'Open 大众点评')
    mobile_operator.tap_app('大众点评')
    time.sleep(2)
    logging.info(f'Open 腾讯视频')
    mobile_operator.tap_app('腾讯视频')

if __name__ == '__main__':
    logger.info(f'MobileOperator start')
    mobile_op = MobileOperator()
    logger.info(f'MobileOperator init ok')
    # mobile_op.get_mobile_info()
    # test_home_page(mobile_op)
    #test_screen_view(mobile_op)
    # test_screen_swipe(mobile_op)
    test_get_screen_views(mobile_op)