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
    def __init__(self, name, obj, desc, element_type):
        self.name = name
        self.object = obj
        self.desc = desc
        self.type = type

# 按照页面建立一颗树
class AppBank:
    def __init__(self):
        self.bank = {}
    def push_element(self,screen_idx,element):
        try:
            ele_type = element.get_attribute('class')
        except:
            ele_type = 'no class'

        try:
            ele_text = element.text
        except:
            ele_text = 'no text'

        try:
            ele_tag_name = element.tag_name
        except:
            ele_tag_name = 'no tag name'



        if not ele_text and not ele_tag_name:
            if ele_type == 'android.widget.ImageView':
                print('image should get text by ai')
                return None
            else:
                return None
        print(f'Push element {ele_text}:{ele_tag_name} -- {ele_type}')

        if not element.get_attribute('clickable'):
            return None

        ele_name = element.text if element.text else element.tag_name
        # ele_name = element.text if element.text else element.tag_name

        app_info = AppInfo(ele_name, element, '', ele_type)
        if screen_idx not in self.bank:
            self.bank[screen_idx] = [app_info]
        else:
            self.bank[screen_idx].append(app_info)
        return app_info

    def get_app(self,name):
        for screen_idx in self.bank:
            for app in self.bank[screen_idx]:
                if app.name == name:
                    return app.object
        return None

    def get_screen_apps(self,screen_idx):
        if screen_idx in self.bank:
            return self.bank[screen_idx]
        return []

    def dump_bank(self):
        for screen_idx in self.bank.keys():
            print('################################')
            print(f'List screen {screen_idx} apps:')
            for app in self.bank[screen_idx]:
                print(f'app name:{app.name} - type{app.type}')

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
        self.hotseat_selector = {}
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

    def goto_screen(self, screen_id = 1):
        logger.info(f'Go to home page')
        if screen_id == '0':
            screen_name = '智能助理'
        else:
            screen_name = f'第{screen_id}屏'

        # cost 130ms
        self.driver.press_keycode(3)
        logger.info(f'start to wait for homepage')
        # cost 7ms
        self.driver.implicitly_wait(2)

        logger.info(f'Go to home page over')
        if screen_id == 1:
            return ReturnValue(flag= True)

        try:
            logger.info(f'Go to screen {screen_name}')
            # cost 130ms
            screen_element = self.driver.find_element(by=AppiumBy.ACCESSIBILITY_ID,value = screen_name)
            logger.info(f'find screen {screen_name}')
            # cost 50ms
            screen_element.click()
            logger.info(f'Go to screen {screen_name} cmd over')
        except NoSuchElementException as e:
            # cost 2.3s
            logger.info(f'No screen {screen_name} found')
            return ReturnValue(flag = False,message='No screen named {screen_name}')

        return ReturnValue(flag = True)

    def swipe(self, left = True):
        # if screen == 0:
        #     self.driver.press_keycode(3)
        #     self.driver.implicitly_wait(10)
        end_pos = self.width
        if left:
            end_pos = 0
        logger.info(f'Swipe to left {left} width {self.width} to {end_pos}')
        self.driver.swipe(self.width - 10, self.height / 2,
                          end_pos, self.height / 2, 500)

    def init_app_bank(self):
        elements = self.get_elements()
        for ele in elements:
            self.app_bank.push_element(0,ele)

    def get_elements(self, value="//*[@clickable='true']"):
        elements = self.driver.find_elements(by=AppiumBy.XPATH,value = value)
        return elements

    def function_key(self,key = AndroidKey.HOME):
        self.driver.press_keycode(int(key))
        return ReturnValue(flag=True)

    def goto_screen_view(self, screen_id: int = 0):
        if screen_id >= len(self.screen_selector):
            return
        try:
            self.screen_selector[screen_id].click()
        except StaleElementReferenceException as e:
            self.driver.press_keycode(3)
            self.screen_selector[screen_id].click()

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

    def get_screen_view(self,refresh = False):
        # screen_views = self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='com.miui.home.launcher.ScreenView')

        # return self.driver.find_elements(by=AppiumBy.ACCESSIBILITY_ID,value=f'第*屏')
        logger.info(f'Get screen view start')
        if len(self.screen_views) != 0 and refresh == False:
            return  self.screen_views

        self.screen_views['智能助理'] = self.driver.find_element(by=AppiumBy.ACCESSIBILITY_ID, value='智能助理')
        logger.info(f'Get {value} finish')
        for idx in range(1,3):
            try:
                views = self.driver.find_element(by=AppiumBy.ACCESSIBILITY_ID,value=f'第{idx}屏')
                if not views:
                    break
            except:
                break
            self.screen_views[f'第{idx}屏'] = views
        # print(f'get_screen_view count {len(self.screen_views)}')
        logger.info(f'Get {len(self.screen_views)} sceens')
        return  self.screen_views

    def get_all_apps_ex(self):
        for idx in range(1,len(self.screen_selector)):
            self.goto_screen(idx)
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
                logger.info(f'app {app.tag_name}')


    def get_all_apps(self):
        screen_id = 0

        for idx,screen in self.screen_views.items():
            logger.info(f'swip to screen {idx}')
            if screen_id == 0:
                screen.click()

            self.driver.implicitly_wait(2)

            # logger.info(f'Get screen {screen.text}: {screen.tag_name} start')
            screen_elements = self.get_elements()
            logger.info(f'Get screen: {idx} apps count {len(screen_elements)}')

            for ele in screen_elements:
                app = self.app_bank.push_element(idx,ele)
                if not app:
                    continue
                if app.name not in self.all_app:
                    self.all_app[app.name] = app

            if screen_id < len(self.screen_views):
                self.swipe()
                screen_id += 1

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
    mobile_operator.get_all_apps_ex()


if __name__ == '__main__':
    logger.info(f'MobileOperator start')
    mobile_op = MobileOperator()
    logger.info(f'MobileOperator init ok')
    # mobile_op.get_mobile_info()
    # test_home_page(mobile_op)
    #test_screen_view(mobile_op)
    # test_screen_swipe(mobile_op)
    test_get_screen_views(mobile_op)